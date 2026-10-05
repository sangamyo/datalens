"""Build a train/val export of a dataset's (filtered) samples as a zip of JSONL files.

Zip layout:
    manifest.json   source, filter, format, counts, created_at
    train.jsonl     one object per line
    val.jsonl       one object per line (may be empty)

Formats:
    jsonl -> {"prompt", "context", "response", "category"}
    chat  -> {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} (context appended to the user turn)

The split is deterministic: sample ids are shuffled with random.Random(export_id).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app import storage
from app.db import SessionLocal
from app.models import Dataset, Export, Sample
from app.queries import apply_sample_filter
from app.schemas import SampleFilter

log = logging.getLogger(__name__)

GENERATOR = "DataLens"
STREAM_BATCH = 1000


class ExportError(Exception):
    """Expected failure with a user-facing message."""


# ---------- pure helpers ----------
def val_count(n: int, val_ratio: float) -> int:
    """n * ratio rounded half-up (same as the UI's Math.round), at least 1 when n >= 2 and ratio > 0,
    and always leaving >= 1 train sample."""
    if n <= 0 or val_ratio <= 0:
        return 0
    k = math.floor(n * val_ratio + 0.5)
    if n >= 2:
        k = max(1, k)
    return max(0, min(k, n - 1))


def split_ids(ids: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic split: shuffle sorted ids with Random(seed); the first k become val.
    Returns (train_ids, val_ids), each sorted."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    k = val_count(len(order), val_ratio)
    return sorted(order[k:]), sorted(order[:k])


def format_record(prompt: str, context: str | None, response: str, category: str | None, fmt: str) -> dict:
    if fmt == "chat":
        user = prompt or ""
        if context:
            user = f"{user}\n\n{context}"
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": response or ""}]}
    if fmt == "jsonl":
        return {"prompt": prompt or "", "context": context, "response": response or "", "category": category}
    raise ExportError(f"unknown export format: {fmt}")


def format_line(prompt: str, context: str | None, response: str, category: str | None, fmt: str) -> str:
    return json.dumps(format_record(prompt, context, response, category, fmt), ensure_ascii=False) + "\n"


def build_manifest(
    *,
    export_id: int,
    hf_repo_id: str,
    config: str | None,
    split: str | None,
    revision: str | None,
    fields: dict | None,
    filter: dict,
    fmt: str,
    val_ratio: float,
    num_train: int,
    num_val: int,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "generator": GENERATOR,
        "export_id": export_id,
        "source": {"hf_repo_id": hf_repo_id, "config": config, "split": split, "revision": revision},
        "fields": fields or {},
        "filter": filter,
        "format": fmt,
        "val_ratio": val_ratio,
        "counts": {"total": num_train + num_val, "train": num_train, "val": num_val},
        "files": {"train": "train.jsonl", "val": "val.jsonl"},
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
    }


def write_export_zip(path: str, manifest: dict, train_path: str, val_path: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
        zf.write(train_path, "train.jsonl")
        zf.write(val_path, "val.jsonl")


# ---------- job ----------
def _set_status(export_id: int, **fields: Any) -> None:
    with SessionLocal() as db:
        export = db.get(Export, export_id)
        if export is None:
            return
        for k, v in fields.items():
            setattr(export, k, v)
        db.commit()


def _build(export_id: int) -> int:
    workdir = tempfile.mkdtemp(prefix=f"export-{export_id}-")
    try:
        with SessionLocal() as db:
            export = db.get(Export, export_id)
            if export is None:
                raise ExportError(f"export {export_id} not found")
            dataset = db.get(Dataset, export.dataset_id)
            if dataset is None:
                raise ExportError(f"dataset {export.dataset_id} not found")
            export.status = "building"
            export.error = None
            db.commit()

            stored = {k: v for k, v in (export.filter or {}).items() if k != "dataset_id"}
            f = SampleFilter(**stored, dataset_id=export.dataset_id)
            fmt = export.format or "jsonl"
            format_record("", None, "", None, fmt)  # validate the format before doing any work

            ids = list(db.scalars(apply_sample_filter(select(Sample.id), f)).all())
            if not ids:
                raise ExportError("no samples match the filter")
            _, val_ids = split_ids(ids, export.val_ratio, export_id)
            val_set = set(val_ids)

            train_path = os.path.join(workdir, "train.jsonl")
            val_path = os.path.join(workdir, "val.jsonl")
            n_train = n_val = 0
            stmt = (
                apply_sample_filter(
                    select(Sample.id, Sample.prompt, Sample.context, Sample.response, Sample.category), f
                )
                .order_by(Sample.sample_index)
                .execution_options(yield_per=STREAM_BATCH)
            )
            with open(train_path, "w", encoding="utf-8") as tf, open(val_path, "w", encoding="utf-8") as vf:
                for sid, prompt, context, response, category in db.execute(stmt):
                    line = format_line(prompt, context, response, category, fmt)
                    if sid in val_set:
                        vf.write(line)
                        n_val += 1
                    else:
                        tf.write(line)
                        n_train += 1
            if n_train + n_val == 0:  # rows deleted between the two queries
                raise ExportError("no samples match the filter")

            manifest = build_manifest(
                export_id=export_id,
                hf_repo_id=dataset.hf_repo_id,
                config=dataset.config,
                split=dataset.split,
                revision=dataset.revision,
                fields=dataset.fields,
                filter=f.model_dump(exclude_none=True),
                fmt=fmt,
                val_ratio=export.val_ratio,
                num_train=n_train,
                num_val=n_val,
            )

        zip_path = os.path.join(workdir, f"export-{export_id}.zip")
        write_export_zip(zip_path, manifest, train_path, val_path)
        key = f"exports/{export_id}.zip"
        storage.put_file(key, zip_path, content_type="application/zip")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    total = n_train + n_val
    _set_status(export_id, status="ready", s3_key=key, num_samples=total, error=None)
    return total


async def build_export(ctx: dict[str, Any], export_id: int) -> None:
    try:
        n = await asyncio.to_thread(_build, export_id)
        log.info("export %s ready with %s samples", export_id, n)
    except ExportError as exc:
        log.warning("export %s failed: %s", export_id, exc)
        await asyncio.to_thread(_set_status, export_id, status="failed", error=str(exc), num_samples=0)
    except Exception as exc:
        log.exception("export %s failed", export_id)
        await asyncio.to_thread(
            _set_status, export_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:2000], num_samples=0
        )

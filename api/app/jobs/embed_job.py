"""arq job: embed every sample of a dataset (prompt + response) and store the vectors in `embeddings`.

Enqueued after every import and by `POST /datasets/{id}/embeddings`. Incremental and idempotent:
only samples without a vector for the current model are embedded, vectors of another model are
dropped first, and each batch is committed on its own, so a killed worker resumes where it stopped.
Rows are inserted with ON CONFLICT (sample_id) DO NOTHING, so two overlapping runs cannot collide.

Texts are embedded shortest first. The model pads every batch to its longest text, so one 2,000-char
row in a batch of 32 short ones makes all 32 cost 512 tokens; sorting by length removes almost all
padding (3-7x faster on dolly-15k, see README).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import delete, exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app import embeddings
from app.db import SessionLocal
from app.models import Dataset, Embedding, Sample

log = logging.getLogger(__name__)

WRITE_BATCH = 512  # samples embedded and committed per round


def _insert_ignoring_existing(db: Session, rows: list[dict[str, Any]]) -> None:
    insert = pg_insert if db.get_bind().dialect.name == "postgresql" else sqlite_insert  # sqlite: tests
    db.execute(insert(Embedding).on_conflict_do_nothing(index_elements=["sample_id"]), rows)


def embed_dataset_sync(dataset_id: int) -> dict[str, Any] | None:
    t0 = time.perf_counter()
    model = embeddings.MODEL_NAME
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            log.warning("embed: dataset %s not found", dataset_id)
            return None
        in_dataset = select(Sample.id).where(Sample.dataset_id == dataset_id)
        db.execute(delete(Embedding).where(Embedding.sample_id.in_(in_dataset), Embedding.model != model))
        db.commit()
        has_vector = exists().where(Embedding.sample_id == Sample.id, Embedding.model == model)
        rows = db.execute(
            select(Sample.id, Sample.prompt, Sample.response).where(Sample.dataset_id == dataset_id, ~has_vector)
        ).all()
    todo = sorted(((r.id, embeddings.sample_text(r.prompt, r.response)) for r in rows), key=lambda t: (len(t[1]), t[0]))

    t_embed = 0.0
    with SessionLocal() as db:
        for start in range(0, len(todo), WRITE_BATCH):
            batch = todo[start : start + WRITE_BATCH]
            t = time.perf_counter()
            vectors = embeddings.embed_texts([text for _, text in batch])
            t_embed += time.perf_counter() - t
            # Samples deleted meanwhile (dataset re-import/delete) would violate the FK: skip them.
            alive = set(db.scalars(select(Sample.id).where(Sample.id.in_([sid for sid, _ in batch]))))
            new = [{"sample_id": sid, "model": model, "embedding": v} for (sid, _), v in zip(batch, vectors) if sid in alive]
            if new:
                _insert_ignoring_existing(db, new)
            db.commit()

    seconds = time.perf_counter() - t0
    log.info(
        "embed: dataset %s, %d samples embedded with %s in %.1fs (model %.1fs)",
        dataset_id, len(todo), model, seconds, t_embed,
    )
    return {"dataset_id": dataset_id, "embedded": len(todo), "seconds": round(seconds, 2)}


async def embed_dataset(ctx: dict[str, Any] | None, dataset_id: int) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(embed_dataset_sync, dataset_id)
    except Exception as exc:  # already-committed batches are kept; a re-run embeds the rest
        log.exception("embed: dataset %s failed", dataset_id)
        return {"dataset_id": dataset_id, "error": f"{type(exc).__name__}: {exc}"[:500]}

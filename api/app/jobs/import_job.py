"""Import an instruction / chat dataset from the Hugging Face Hub (arq job).

Source files, in order of preference:

1. The Hub's auto-converted parquet (revision ``refs/convert/parquet``):
   ``{config}/{split}/*.parquet`` (or ``{config}/partial-{split}/*.parquet`` for very large
   splits, or the older flat ``{config}/{split}-*.parquet`` naming).
2. Data files on the main revision: ``*.parquet``, then ``*.jsonl``, then ``*.json``, picked by
   config directory and split name (``data/train-0000-of-0001.parquet``, ``main/train-*.parquet``,
   ``train.jsonl``...). A repo with a single data file and no split names in it (e.g. dolly's
   ``databricks-dolly-15k.jsonl``) is treated as its ``train`` split.

Only the first ``max_samples`` rows are read (files are opened one at a time, row-batch by
row-batch, and we stop as soon as we have enough). Every file we read rows from is copied to
object storage as ``datasets/{id}/raw/{n}.parquet`` (JSON/JSONL sources are converted to parquet
first, containing the rows read).

Field mapping is auto-detected from the column names (see ``detect_fields``) unless an explicit
mapping is passed. The pure helpers at the top of this module are unit-tested without network
or database access.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from sqlalchemy import delete, insert
from sqlalchemy.exc import IntegrityError

from app import storage
from app.config import get_settings
from app.db import SessionLocal
from app.models import Dataset, Sample

log = logging.getLogger(__name__)

MAX_ERROR_LEN = 2000
INSERT_BATCH = 1000
CATEGORY_MAX_LEN = 100  # Sample.category is String(100)
CONVERT_REVISION = "refs/convert/parquet"
MAX_FOOTERS = 200  # parquet files whose footer we read to count total rows

PROMPT_COLS = ("instruction", "prompt", "question", "query", "input")
CONTEXT_COLS = ("context", "input")
RESPONSE_COLS = ("response", "output", "answer", "completion", "chosen")
CATEGORY_COLS = ("category", "task", "type", "label", "source")
CHAT_COLS = ("messages", "conversations")

USER_ROLES = {"user", "human", "prompter"}
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatgpt"}
SYSTEM_ROLES = {"system"}
SPLIT_NAMES = ("train", "test", "validation", "valid", "val", "dev", "eval")
DATA_EXTS = (".parquet", ".jsonl", ".json")


class DatasetImportError(ValueError):
    """A user-facing import failure (bad config/split, no usable columns...)."""


# ======================================================================== field detection
def _is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)


def _is_scalar(t: pa.DataType) -> bool:
    return _is_text(t) or pa.types.is_integer(t) or pa.types.is_boolean(t) or pa.types.is_dictionary(t)


def _find(schema: pa.Schema, candidates: Iterable[str], ok, exclude: set[str] = frozenset()) -> str | None:
    """First candidate name present in `schema` (exact name first, then case-insensitive) whose type passes `ok`."""
    by_lower: dict[str, str] = {}
    for name in schema.names:
        by_lower.setdefault(name.lower(), name)
    for cand in candidates:
        for name in (cand, by_lower.get(cand.lower())):
            if name and name in schema.names and name not in exclude and ok(schema.field(name).type):
                return name
    return None


def detect_fields(schema: pa.Schema, explicit: dict[str, Any] | None = None) -> dict[str, str]:
    """Return the column mapping to use, e.g. {"prompt": "instruction", "context": "input", "response": "output"}
    or {"messages": "messages", "category": "category"} for chat datasets.

    Raises DatasetImportError listing the available columns when nothing usable is found."""
    available = ", ".join(schema.names) or "(none)"
    explicit = {k: v for k, v in (explicit or {}).items() if v}
    if explicit:
        missing = [v for v in explicit.values() if v not in schema.names]
        if missing:
            raise DatasetImportError(f"column(s) not found: {', '.join(missing)}; available columns: {available}")
        if "messages" not in explicit and not ("prompt" in explicit and "response" in explicit):
            raise DatasetImportError(
                f"explicit fields need prompt+response or messages; available columns: {available}"
            )
        return {k: explicit[k] for k in ("prompt", "context", "response", "category", "messages") if k in explicit}

    prompt = _find(schema, PROMPT_COLS, _is_text)
    response = _find(schema, RESPONSE_COLS, _is_text, exclude={prompt} if prompt else set())
    if prompt and response:
        out = {"prompt": prompt}
        # A context column only makes sense next to a separate instruction column
        # (alpaca: instruction + input; when `input` is the prompt there is no context).
        context = _find(schema, CONTEXT_COLS, _is_text, exclude={prompt, response})
        if context:
            out["context"] = context
        out["response"] = response
    else:
        chat = _find(schema, CHAT_COLS, _is_list)
        if not chat:
            raise DatasetImportError(f"no prompt/response columns found; available columns: {available}")
        out = {"messages": chat}
    used = set(out.values())
    category = _find(schema, CATEGORY_COLS, _is_scalar, exclude=used)
    if category:
        out["category"] = category
    return out


def class_label_names(schema: pa.Schema) -> dict[str, list[str]]:
    """ClassLabel columns are stored as ints; their names live in the HF schema metadata."""
    raw = (schema.metadata or {}).get(b"huggingface")
    if not raw:
        return {}
    try:
        feats = json.loads(raw).get("info", {}).get("features", {})
    except (ValueError, AttributeError):
        return {}
    return {
        col: list(f["names"])
        for col, f in feats.items()
        if isinstance(f, dict) and f.get("_type") == "ClassLabel" and isinstance(f.get("names"), list)
    }


# ======================================================================== text helpers
def norm_text(value: Any) -> str:
    """Coerce a cell to a stripped string ("" for None). Postgres text cannot hold NUL bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    elif not isinstance(value, str):
        value = str(value)
    return value.replace("\x00", "").strip()


def _content_text(content: Any) -> str:
    """Message content may be a string or a list of parts ([{"type": "text", "text": ...}])."""
    if isinstance(content, list):
        parts = [p.get("text") if isinstance(p, dict) else p for p in content]
        return norm_text("\n".join(norm_text(p) for p in parts if p))
    return norm_text(content)


def parse_chat(messages: Any) -> tuple[str, str, str | None]:
    """(first user turn, first assistant turn after it, system prompt or None).

    Accepts [{role, content}] (OpenAI / HF chat template) and [{from, value}] (ShareGPT)."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return norm_text(messages), "", None
    prompt: str | None = None
    response: str | None = None
    system: str | None = None
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = norm_text(m.get("role", m.get("from"))).lower()
        content = _content_text(m.get("content", m.get("value")))
        if role in SYSTEM_ROLES and system is None and prompt is None:
            system = content or None
        elif role in USER_ROLES and prompt is None:
            prompt = content
        elif role in ASSISTANT_ROLES and prompt is not None and response is None:
            response = content
            break
    return prompt or "", response or "", system


def _norm_for_hash(s: str) -> str:
    return " ".join(s.lower().split())


def content_hash(prompt: str, context: str | None, response: str) -> str:
    """sha1 of whitespace-normalised, lower-cased prompt + "\\n" + context + "\\n" + response."""
    payload = "\n".join(_norm_for_hash(p) for p in (prompt, context or "", response))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def estimate_tokens(*texts: str | None) -> int:
    return math.ceil(sum(len(t or "") for t in texts) / 4)


def build_sample(dataset_id: int, index: int, prompt: Any, context: Any, response: Any, category: Any) -> dict:
    """A row for `samples` (a dict for bulk insert) with derived columns filled in."""
    p, r = norm_text(prompt), norm_text(response)
    c = norm_text(context) or None
    cat = norm_text(category)[:CATEGORY_MAX_LEN] or None
    return {
        "dataset_id": dataset_id,
        "sample_index": index,
        "prompt": p,
        "context": c,
        "response": r,
        "category": cat,
        "prompt_chars": len(p),
        "response_chars": len(r),
        "tokens_est": estimate_tokens(p, c, r),
        "content_hash": content_hash(p, c, r),
        "qc_status": "pending",
    }


def extract_row(row: dict[str, Any], fields: dict[str, str], labels: dict[str, list[str]] | None = None):
    """(prompt, context, response, category) for one source row under the given mapping."""
    labels = labels or {}
    cat_col = fields.get("category")
    category = row.get(cat_col) if cat_col else None
    if cat_col in labels and isinstance(category, int) and 0 <= category < len(labels[cat_col]):
        category = labels[cat_col][category]
    if "messages" in fields:
        prompt, response, system = parse_chat(row.get(fields["messages"]))
        # An explicit prompt/response column still wins over the chat turns.
        if fields.get("prompt"):
            prompt = row.get(fields["prompt"])
        if fields.get("response"):
            response = row.get(fields["response"])
        context = row.get(fields["context"]) if fields.get("context") else system
        return prompt, context, response, category
    ctx_col = fields.get("context")
    return row.get(fields["prompt"]), row.get(ctx_col) if ctx_col else None, row.get(fields["response"]), category


# ======================================================================== file selection
def convert_configs(files: Iterable[str]) -> list[str]:
    """Config names in a refs/convert/parquet listing (top-level directories holding parquet)."""
    seen: dict[str, None] = {}
    for f in files:
        if f.endswith(".parquet") and "/" in f:
            seen.setdefault(f.split("/", 1)[0], None)
    return list(seen)


def convert_splits(files: Iterable[str], config: str) -> list[str]:
    splits: dict[str, None] = {}
    for f in files:
        parts = f.split("/")
        if parts[0] != config or not f.endswith(".parquet"):
            continue
        if len(parts) >= 3:
            name = parts[1].removeprefix("partial-")
        else:  # old flat layout: {config}/{split}-00000-of-00001.parquet
            name = re.split(r"-\d", parts[1].removesuffix(".parquet"), maxsplit=1)[0]
        splits.setdefault(name, None)
    return list(splits)


def select_convert_files(files: Iterable[str], config: str, split: str) -> list[str]:
    files = [f for f in files if f.endswith(".parquet")]
    for prefix in (f"{config}/{split}/", f"{config}/partial-{split}/"):
        hits = sorted(f for f in files if f.startswith(prefix))
        if hits:
            return hits
    flat = re.compile(rf"^{re.escape(config)}/{re.escape(split)}(-[^/]*)?\.parquet$")
    return sorted(f for f in files if flat.match(f))


def pick_default_config(configs: list[str], card_configs: list[tuple[str, bool]] | None = None) -> str:
    """'default' if present, else the dataset card's default config, else the first one."""
    card_configs = card_configs or []
    names = list(configs) or [name for name, _ in card_configs]
    if not names or "default" in names:
        return "default"
    for name, is_default in card_configs:
        if is_default and name in names:
            return name
    for name, _ in card_configs:
        if name in names:
            return name
    return names[0]


def _split_pattern(split: str) -> re.Pattern:
    # matches "train.jsonl", "data/train-0000-of-0001.parquet", "main/train/x.parquet"; not "train_sft-...".
    return re.compile(rf"(^|/){re.escape(split)}([-./]|$)")


def select_main_files(files: Iterable[str], config: str, split: str) -> list[str]:
    """Data files on the main revision for config/split, all of one format (parquet > jsonl > json)."""
    data = [f for f in files if f.lower().endswith(DATA_EXTS) and not f.split("/")[-1].startswith(".")]
    if config != "default":
        data = [f for f in data if config in f.split("/")[:-1]]
    pat = _split_pattern(split)
    chosen = [f for f in data if pat.search(f)]
    if not chosen and split == "train":
        # No split names anywhere (e.g. "databricks-dolly-15k.jsonl"): everything is train.
        if not any(_split_pattern(s).search(f) for s in SPLIT_NAMES for f in data):
            chosen = data
    for ext in DATA_EXTS:
        of_kind = sorted(f for f in chosen if f.lower().endswith(ext))
        if of_kind:
            return of_kind
    return []


# ======================================================================== reading
class _Reader:
    """Accumulates sample rows from successive pyarrow batches until max_samples is reached."""

    def __init__(self, dataset_id: int, max_samples: int, explicit_fields: dict | None):
        self.dataset_id = dataset_id
        self.max_samples = max_samples
        self.explicit = explicit_fields
        self.fields: dict[str, str] | None = None
        self.labels: dict[str, list[str]] = {}
        self.samples: list[dict] = []

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.max_samples

    def set_schema(self, schema: pa.Schema) -> None:
        if self.fields is None:
            self.fields = detect_fields(schema, self.explicit)
            self.labels = class_label_names(schema)

    def add_batch(self, batch: pa.RecordBatch | pa.Table) -> None:
        assert self.fields is not None
        cols = [c for c in dict.fromkeys(self.fields.values()) if c in batch.schema.names]
        need = self.max_samples - len(self.samples)
        rows = batch.select(cols).slice(0, need).to_pylist()
        for row in rows:
            idx = len(self.samples)
            self.samples.append(build_sample(self.dataset_id, idx, *extract_row(row, self.fields, self.labels)))


def _read_json_rows(path: str, kind: str, limit: int) -> tuple[list[dict], int]:
    """(first `limit` rows, total row count) of a .jsonl / .json file."""
    with open(path, encoding="utf-8") as fh:
        head = fh.read(1)
        while head and head.isspace():
            head = fh.read(1)
        fh.seek(0)
        if kind == ".json" and head == "[":
            data = json.load(fh)
            return [r for r in data[:limit] if isinstance(r, dict)], len(data)
        if kind == ".json" and head == "{":
            # Either JSON lines with a .json extension, or {"data": [...]} style.
            try:
                obj = json.load(fh)
            except ValueError:
                fh.seek(0)
            else:
                lists = [v for v in obj.values() if isinstance(v, list)] if isinstance(obj, dict) else []
                data = lists[0] if lists else [obj]
                return [r for r in data[:limit] if isinstance(r, dict)], len(data)
        rows, total = [], 0
        for line in fh:
            if not line.strip():
                continue
            total += 1
            if len(rows) < limit:
                rows.append(json.loads(line))
        return rows, total


@dataclass
class Source:
    config: str
    revision: str  # revision files are downloaded from (a commit sha when known)
    files: list[str]
    from_convert: bool


def _card_configs(info: Any) -> list[tuple[str, bool]]:
    card = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    out = []
    for c in card.get("configs") or []:
        if isinstance(c, dict) and c.get("config_name"):
            out.append((str(c["config_name"]), bool(c.get("default"))))
    return out


def _convert_listing(api: Any, repo_id: str) -> tuple[str, list[str]] | None:
    """(revision, files) of the auto-converted parquet branch, or None if it does not exist."""
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        rev = next((r.target_commit for r in refs.converts if r.ref == CONVERT_REVISION), None)
        if rev is None:
            return None
        return rev, api.list_repo_files(repo_id, repo_type="dataset", revision=rev)
    except Exception as exc:  # no convert branch, or not allowed to list refs
        log.info("no %s for %s: %s", CONVERT_REVISION, repo_id, exc)
        return None


def resolve_default_config(repo_id: str) -> str:
    """The config a request without `config` means (used by the API for the duplicate check)."""
    api = HfApi(token=get_settings().hf_token or None)
    info = api.dataset_info(repo_id)
    conv = _convert_listing(api, repo_id)
    return pick_default_config(convert_configs(conv[1]) if conv else [], _card_configs(info))


def resolve_source(api: Any, repo_id: str, info: Any, config: str, split: str) -> Source:
    conv = _convert_listing(api, repo_id)
    card = _card_configs(info)
    if conv:
        rev, files = conv
        configs = convert_configs(files)
        cfg = config if config in configs else (pick_default_config(configs, card) if config == "default" else None)
        if cfg is None:
            raise DatasetImportError(f"config {config!r} not found; available configs: {', '.join(configs)}")
        hits = select_convert_files(files, cfg, split)
        if hits:
            return Source(cfg, rev, hits, True)
        main_files = info_files(api, repo_id, info.sha)
        fallback = select_main_files(main_files, cfg, split)
        if fallback:
            return Source(cfg, info.sha, fallback, False)
        raise DatasetImportError(
            f"split {split!r} not found in config {cfg!r}; available splits: {', '.join(convert_splits(files, cfg))}"
        )
    main_files = info_files(api, repo_id, info.sha)
    cfg = config
    if config == "default":
        cfg = pick_default_config([], card)
    hits = select_main_files(main_files, cfg, split)
    if not hits and cfg != "default":
        hits = select_main_files(main_files, "default", split)
    if not hits:
        raise DatasetImportError(
            f"no parquet/jsonl/json files for config {cfg!r} split {split!r} "
            f"(no auto-converted parquet either); repo files: {', '.join(main_files[:30])}"
        )
    return Source(cfg, info.sha, hits, False)


def info_files(api: Any, repo_id: str, revision: str) -> list[str]:
    return api.list_repo_files(repo_id, repo_type="dataset", revision=revision)


def _parquet_rows_remote(repo_id: str, revision: str, path: str, token: str | None) -> int | None:
    """Row count from a remote parquet footer (a couple of range requests, no full download)."""
    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{repo_id}@{revision}/{path}", "rb") as fh:
        return pq.ParquetFile(fh).metadata.num_rows


def read_source(
    repo_id: str, src: Source, dataset_id: int, max_samples: int, fields: dict | None, token: str | None
) -> tuple[_Reader, int | None]:
    """Download source files one by one until max_samples rows are collected.
    Uploads each file read to datasets/{id}/raw/{n}.parquet. Returns (reader, total rows or None)."""
    reader = _Reader(dataset_id, max_samples, fields)
    total: int | None = 0
    unread: list[str] = []
    n = 0
    for path in src.files:
        if reader.done:
            unread.append(path)
            continue
        local = hf_hub_download(repo_id, path, repo_type="dataset", revision=src.revision, token=token)
        raw_key = f"datasets/{dataset_id}/raw/{n}.parquet"
        if path.lower().endswith(".parquet"):
            pf = pq.ParquetFile(local)
            total += pf.metadata.num_rows
            reader.set_schema(pf.schema_arrow)
            for batch in pf.iter_batches(batch_size=1024):
                reader.add_batch(batch)
                if reader.done:
                    break
            storage.put_file(raw_key, local, "application/vnd.apache.parquet")
        else:
            ext = ".jsonl" if path.lower().endswith(".jsonl") else ".json"
            rows, count = _read_json_rows(local, ext, max_samples - len(reader.samples))
            total += count
            table = pa.Table.from_pylist(rows)
            reader.set_schema(table.schema)
            reader.add_batch(table)
            buf = io.BytesIO()
            pq.write_table(table, buf)
            storage.put_bytes(raw_key, buf.getvalue(), "application/vnd.apache.parquet")
        n += 1
    if unread:
        if not all(p.lower().endswith(".parquet") for p in unread) or len(unread) > MAX_FOOTERS:
            return reader, None
        try:
            for path in unread:
                total += _parquet_rows_remote(repo_id, src.revision, path, token) or 0
        except Exception as exc:
            log.warning("could not count rows of %s: %s", repo_id, exc)
            total = None
    if reader.fields is None:
        raise DatasetImportError("no rows found in the source files")
    return reader, total


# ======================================================================== job
def _set_status(dataset_id: int, **values: Any) -> None:
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            return
        for k, v in values.items():
            setattr(ds, k, v)
        db.commit()


def _insert_samples(db, samples: list[dict]) -> None:
    for i in range(0, len(samples), INSERT_BATCH):
        db.execute(insert(Sample), samples[i : i + INSERT_BATCH])


def _import_sync(dataset_id: int, max_samples: int, fields: dict | None) -> int:
    token = get_settings().hf_token or None
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise LookupError(f"dataset {dataset_id} not found")
        repo_id, config, split = ds.hf_repo_id, ds.config or "default", ds.split or "train"
        # Idempotent re-run: start from a clean slate (QC results/embeddings cascade).
        db.execute(delete(Sample).where(Sample.dataset_id == dataset_id))
        ds.status, ds.error, ds.num_samples = "importing", None, 0
        db.commit()

    storage.ensure_bucket()
    storage.delete_prefix(f"datasets/{dataset_id}/raw/")

    api = HfApi(token=token)
    info = api.dataset_info(repo_id)
    src = resolve_source(api, repo_id, info, config, split)
    log.info(
        "importing %s config=%s split=%s from %s (%d file(s), max %d rows)",
        repo_id, src.config, split, "convert/parquet" if src.from_convert else "main", len(src.files), max_samples,
    )
    reader, total = read_source(repo_id, src, dataset_id, max_samples, fields, token)

    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise LookupError(f"dataset {dataset_id} was deleted during import")
        if src.config != ds.config:
            ds.config = src.config
            try:
                db.flush()
            except IntegrityError:
                raise DatasetImportError(f"{repo_id} config {src.config!r} split {split!r} is already imported") from None
        _insert_samples(db, reader.samples)
        ds.name = repo_id.split("/")[-1]
        ds.revision = info.sha
        ds.fields = reader.fields
        ds.num_samples = len(reader.samples)
        ds.total_samples = total
        ds.status = "ready"
        ds.error = None
        db.commit()
    return len(reader.samples)


async def import_dataset(
    ctx: dict[str, Any], dataset_id: int, max_samples: int = 2000, fields: dict | None = None
) -> int:
    try:
        n = await asyncio.to_thread(_import_sync, dataset_id, max_samples, fields)
    except Exception as exc:
        log.exception("import of dataset %s failed", dataset_id)
        msg = str(exc) if isinstance(exc, DatasetImportError) else f"{type(exc).__name__}: {exc}"
        await asyncio.to_thread(_set_status, dataset_id, status="failed", error=msg[:MAX_ERROR_LEN])
        return 0
    await ctx["redis"].enqueue_job("run_dataset_qc", dataset_id)
    await ctx["redis"].enqueue_job("embed_dataset", dataset_id)
    log.info("dataset %s ready: %d samples, QC and embeddings enqueued", dataset_id, n)
    return n

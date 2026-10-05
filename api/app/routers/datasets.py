import asyncio
import logging
import math

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import storage
from app.db import get_db
from app.jobs.import_job import resolve_default_config
from app.models import Dataset, Export, Sample
from app.queue import enqueue
from app.schemas import CategoryCount, DatasetDetail, DatasetOut, HistogramBucket, ImportRequest, QCCounts

router = APIRouter(prefix="/datasets", tags=["datasets"])
log = logging.getLogger(__name__)

UNCATEGORISED = "uncategorised"
MAX_CATEGORIES = 30
HISTOGRAM_BUCKETS = 12
CONFIG_LOOKUP_TIMEOUT_S = 15


def _get_or_404(db: Session, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    return ds


# ---------------------------------------------------------------- pure helpers (unit-tested)
def percentile_nearest_rank(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of an ascending list (q in 0..100)."""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def token_histogram(values: list[int], buckets: int = HISTOGRAM_BUCKETS) -> list[HistogramBucket]:
    """Equal-width buckets from min to p99; the last bucket is open-ended (its `end` is the max value)
    so outliers above p99 are counted there instead of stretching every bucket."""
    if not values:
        return []
    vals = sorted(values)
    lo, hi, top = vals[0], percentile_nearest_rank(vals, 99), vals[-1]
    if hi <= lo:
        return [HistogramBucket(start=lo, end=top, count=len(vals))]
    width = max(1, math.ceil((hi - lo) / buckets))
    n = min(buckets, math.ceil((hi - lo) / width))
    counts = [0] * n
    for v in vals:
        counts[min((v - lo) // width, n - 1)] += 1
    out = [HistogramBucket(start=lo + i * width, end=lo + (i + 1) * width, count=c) for i, c in enumerate(counts)]
    out[-1].end = max(out[-1].end, top)
    return out


# ---------------------------------------------------------------- queries
def qc_counts(db: Session, dataset_id: int) -> QCCounts:
    rows = db.execute(
        select(Sample.qc_status, func.count()).where(Sample.dataset_id == dataset_id).group_by(Sample.qc_status)
    ).all()
    return QCCounts.model_validate({s: n for s, n in rows if s in ("pending", "pass", "warn", "fail")})


def category_counts(db: Session, dataset_id: int, limit: int = MAX_CATEGORIES) -> list[CategoryCount]:
    name = func.coalesce(Sample.category, UNCATEGORISED)
    n = func.count().label("n")
    rows = db.execute(
        select(name, n).where(Sample.dataset_id == dataset_id).group_by(name).order_by(n.desc(), name).limit(limit)
    ).all()
    return [CategoryCount(name=c, count=k) for c, k in rows]


def dataset_histogram(db: Session, dataset_id: int) -> list[HistogramBucket]:
    values = db.scalars(select(Sample.tokens_est).where(Sample.dataset_id == dataset_id)).all()
    return token_histogram(list(values))


# ---------------------------------------------------------------- endpoints
async def _request_config(req: ImportRequest) -> str:
    """Normalise the requested config so the duplicate check matches what the import job stores."""
    if req.config:
        return req.config
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(resolve_default_config, req.hf_repo_id), timeout=CONFIG_LOOKUP_TIMEOUT_S
        )
    except Exception as exc:  # Hub unreachable / repo missing: the job resolves it (and reports errors)
        log.info("could not resolve default config of %s: %s", req.hf_repo_id, exc)
        return "default"


@router.post("/import", status_code=202, response_model=DatasetOut)
async def import_dataset(req: ImportRequest, db: Session = Depends(get_db)) -> Dataset:
    config = await _request_config(req)
    label = f"{req.hf_repo_id} (config {config!r}, split {req.split!r})"
    exists = db.scalar(
        select(Dataset.id).where(
            Dataset.hf_repo_id == req.hf_repo_id, Dataset.config == config, Dataset.split == req.split
        )
    )
    if exists is not None:
        raise HTTPException(status_code=409, detail=f"{label} is already imported (dataset {exists})")
    ds = Dataset(
        hf_repo_id=req.hf_repo_id,
        config=config,
        split=req.split,
        name=req.hf_repo_id.split("/")[-1],
        status="pending",
        num_samples=0,
    )
    db.add(ds)
    try:
        db.commit()
    except IntegrityError:  # lost a race with a concurrent import of the same repo/config/split
        db.rollback()
        raise HTTPException(status_code=409, detail=f"{label} is already imported") from None
    db.refresh(ds)
    fields = req.fields.model_dump(exclude_none=True) if req.fields else None
    try:
        await enqueue("import_dataset", ds.id, req.max_samples, fields or None)
    except Exception as exc:
        ds.status, ds.error = "failed", f"could not enqueue import job: {exc}"[:2000]
        db.commit()
        raise HTTPException(status_code=503, detail="job queue unavailable") from exc
    return ds


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)) -> list[Dataset]:
    return list(db.scalars(select(Dataset).order_by(Dataset.created_at.desc(), Dataset.id.desc())).all())


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> DatasetDetail:
    ds = _get_or_404(db, dataset_id)
    return DatasetDetail(
        **DatasetOut.model_validate(ds).model_dump(),
        qc_counts=qc_counts(db, dataset_id),
        categories=category_counts(db, dataset_id),
        token_histogram=dataset_histogram(db, dataset_id),
    )


def _delete_objects(dataset_id: int, export_keys: list[str]) -> None:
    storage.delete_prefix(f"datasets/{dataset_id}/")
    for key in export_keys:
        storage.delete_prefix(key)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)) -> Response:
    _get_or_404(db, dataset_id)
    exports = db.execute(select(Export.id, Export.s3_key).where(Export.dataset_id == dataset_id)).all()
    export_keys = sorted({key or f"exports/{eid}.zip" for eid, key in exports})
    # FKs are ON DELETE CASCADE: samples, qc_results, embeddings and exports go with it.
    db.execute(delete(Dataset).where(Dataset.id == dataset_id))
    db.commit()
    try:
        await asyncio.to_thread(_delete_objects, dataset_id, export_keys)
    except ClientError as exc:  # e.g. bucket never created: nothing to delete
        log.warning("could not delete objects of dataset %s: %s", dataset_id, exc)
    return Response(status_code=204)

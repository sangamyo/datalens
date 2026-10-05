from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Dataset, QCResult, Sample
from app.queries import find_samples, sample_out
from app.schemas import QCResultOut, QCStatus, SampleDetail, SampleFilter, SampleList

router = APIRouter(tags=["samples"])


@router.get("/datasets/{dataset_id}/samples", response_model=SampleList)
def list_samples(
    dataset_id: int,
    qc_status: list[QCStatus] | None = Query(None),
    category: str | None = None,
    lang: str | None = None,
    min_tokens: int | None = Query(None, ge=0),
    max_tokens: int | None = Query(None, ge=0),
    text_contains: str | None = Query(None, max_length=500),
    failed_check: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> SampleList:
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    f = SampleFilter(
        dataset_id=dataset_id,
        qc_status=qc_status or None,
        category=category or None,
        lang=lang or None,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        text_contains=text_contains or None,
        failed_check=failed_check or None,
    )
    rows, total = find_samples(db, f, limit=limit, offset=offset)
    return SampleList(items=[sample_out(s) for s in rows], total=total)


def neighbour_ids(db: Session, sample: Sample) -> tuple[int | None, int | None]:
    """Ids of the samples with the closest lower / higher sample_index in the same dataset."""
    same = Sample.dataset_id == sample.dataset_id
    prev_id = db.scalar(
        select(Sample.id).where(same, Sample.sample_index < sample.sample_index).order_by(Sample.sample_index.desc()).limit(1)
    )
    next_id = db.scalar(
        select(Sample.id).where(same, Sample.sample_index > sample.sample_index).order_by(Sample.sample_index.asc()).limit(1)
    )
    return prev_id, next_id


@router.get("/samples/{sample_id}", response_model=SampleDetail)
def get_sample(sample_id: int, db: Session = Depends(get_db)) -> SampleDetail:
    s = db.get(Sample, sample_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} not found")
    results = db.scalars(
        select(QCResult).where(QCResult.sample_id == sample_id).order_by(QCResult.check_name, QCResult.id)
    ).all()
    prev_id, next_id = neighbour_ids(db, s)
    return SampleDetail(
        **sample_out(s).model_dump(),
        prompt=s.prompt,
        context=s.context,
        response=s.response,
        prompt_chars=s.prompt_chars,
        response_chars=s.response_chars,
        qc_results=[QCResultOut.model_validate(r) for r in results],
        prev_id=prev_id,
        next_id=next_id,
    )

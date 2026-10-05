"""Quality-check endpoints (see docs/api-contract.md, "Quality checks")."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app import queue
from app.db import get_db
from app.models import Dataset, QCResult, Sample
from app.schemas import CHECK_NAMES, Enqueued, QCCounts, QCResultOut, QCSummary

log = logging.getLogger(__name__)

router = APIRouter(tags=["qc"])


def _dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    return ds


def _sample_or_404(db: Session, sample_id: int) -> Sample:
    s = db.get(Sample, sample_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} not found")
    return s


async def _enqueue(function: str, arg: int) -> None:
    try:
        await queue.enqueue(function, arg)
    except Exception as exc:
        log.exception("qc: enqueue %s(%s) failed", function, arg)
        raise HTTPException(status_code=503, detail="job queue unavailable") from exc


@router.post("/datasets/{dataset_id}/qc", status_code=202, response_model=Enqueued)
async def run_dataset_qc(dataset_id: int, db: Session = Depends(get_db)) -> Enqueued:
    _dataset_or_404(db, dataset_id)
    # Mark pending before enqueueing (row locks make a fast worker's writes land after this commit);
    # roll back if the queue is unavailable.
    db.execute(update(Sample).where(Sample.dataset_id == dataset_id).values(qc_status="pending", qc_score=None))
    try:
        await _enqueue("run_dataset_qc", dataset_id)
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return Enqueued(enqueued=1)


@router.post("/samples/{sample_id}/qc", status_code=202, response_model=Enqueued)
async def run_sample_qc(sample_id: int, db: Session = Depends(get_db)) -> Enqueued:
    _sample_or_404(db, sample_id)
    db.execute(update(Sample).where(Sample.id == sample_id).values(qc_status="pending", qc_score=None))
    try:
        await _enqueue("run_sample_qc", sample_id)
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return Enqueued(enqueued=1)


@router.get("/samples/{sample_id}/qc", response_model=list[QCResultOut])
def get_sample_qc(sample_id: int, db: Session = Depends(get_db)) -> list[QCResult]:
    _sample_or_404(db, sample_id)
    return list(
        db.scalars(select(QCResult).where(QCResult.sample_id == sample_id).order_by(QCResult.check_name, QCResult.id))
    )


@router.get("/datasets/{dataset_id}/qc/summary", response_model=QCSummary)
def get_qc_summary(dataset_id: int, db: Session = Depends(get_db)) -> QCSummary:
    _dataset_or_404(db, dataset_id)
    counts = {"pending": 0, "pass": 0, "warn": 0, "fail": 0}
    for status, n in db.execute(
        select(Sample.qc_status, func.count()).where(Sample.dataset_id == dataset_id).group_by(Sample.qc_status)
    ):
        if status in counts:
            counts[status] += n
    by_check: dict[str, dict[str, int]] = {name: {"warn": 0, "fail": 0} for name in CHECK_NAMES}
    for name, severity, n in db.execute(
        select(QCResult.check_name, QCResult.severity, func.count())
        .join(Sample, Sample.id == QCResult.sample_id)
        .where(Sample.dataset_id == dataset_id, QCResult.severity.in_(["warn", "fail"]))
        .group_by(QCResult.check_name, QCResult.severity)
    ):
        by_check.setdefault(name, {"warn": 0, "fail": 0})[severity] += n
    return QCSummary(counts=QCCounts(**counts), by_check=by_check)

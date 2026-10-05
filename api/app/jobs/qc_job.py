"""arq jobs: run the quality checks for a whole dataset (or one sample) and store the results.

`run_dataset_qc` loads every sample once (streamed in chunks), computes the dataset-wide stats
(length distribution, exact/near-duplicate maps) once, runs all checks for every sample, then
replaces the dataset's qc_results in bulk and updates Sample.qc_status / qc_score / lang in
committed batches. All blocking work runs in a thread (asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Dataset, QCResult, Sample
from app.qc.checks import CheckResult, run_all, summarize
from app.qc.dataset_stats import DatasetStats, SampleRow, compute_stats

log = logging.getLogger(__name__)

LOAD_CHUNK = 2000  # rows fetched per round trip while streaming samples
WRITE_BATCH = 2000  # samples whose results are written per commit


def _error_result(msg: str) -> CheckResult:
    return CheckResult("qc_error", False, "fail", msg[:500], {})


def _exc_message(exc: BaseException) -> str:
    return f"QC failed: {type(exc).__name__}: {exc}".splitlines()[0][:500]


def load_rows(db: Session, dataset_id: int) -> list[SampleRow]:
    stmt = (
        select(
            Sample.id,
            Sample.sample_index,
            Sample.prompt,
            Sample.context,
            Sample.response,
            Sample.tokens_est,
            Sample.content_hash,
        )
        .where(Sample.dataset_id == dataset_id)
        .order_by(Sample.sample_index, Sample.id)
        .execution_options(yield_per=LOAD_CHUNK)
    )
    return [
        SampleRow(
            id=r.id,
            sample_index=r.sample_index,
            prompt=r.prompt or "",
            context=r.context,
            response=r.response or "",
            tokens_est=r.tokens_est or 0,
            content_hash=r.content_hash,
        )
        for r in db.execute(stmt)
    ]


def evaluate(rows: Iterable[SampleRow], stats: DatasetStats) -> dict[int, list[CheckResult]]:
    """Run all checks per sample; an exception in one sample yields a single qc_error result for it."""
    out: dict[int, list[CheckResult]] = {}
    for row in rows:
        try:
            out[row.id] = run_all(row, stats)
        except Exception as exc:  # never let one odd sample kill the dataset run
            log.exception("qc: checks failed for sample %s", row.id)
            out[row.id] = [_error_result(_exc_message(exc))]
    return out


def _lang_of(results: list[CheckResult]) -> str | None:
    for r in results:
        if r.check_name == "non_english":
            return r.details.get("lang")
    return None


def store_results(db: Session, results: dict[int, list[CheckResult]], dataset_id: int | None = None) -> None:
    """Replace qc_results for the given samples and update their status/score/lang, in committed batches.

    With `dataset_id`, all old results of the dataset are deleted in one statement first.
    """
    if dataset_id is not None:
        db.execute(
            delete(QCResult).where(QCResult.sample_id.in_(select(Sample.id).where(Sample.dataset_id == dataset_id)))
        )
    ids = list(results)
    for start in range(0, len(ids), WRITE_BATCH):
        batch = ids[start : start + WRITE_BATCH]
        if dataset_id is None:
            db.execute(delete(QCResult).where(QCResult.sample_id.in_(batch)))
        rows: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        for sid in batch:
            res = results[sid]
            for r in res:
                rows.append(
                    {
                        "sample_id": sid,
                        "check_name": r.check_name,
                        "passed": r.passed,
                        "severity": r.severity,
                        "message": r.message,
                        "details": r.details,
                    }
                )
            status, score = summarize(res)
            upd: dict[str, Any] = {"id": sid, "qc_status": status, "qc_score": score}
            lang = _lang_of(res)
            if lang is not None:
                upd["lang"] = lang
            updates.append(upd)
        if rows:
            db.execute(insert(QCResult), rows)
        # ORM bulk UPDATE by primary key; group by key set so each executemany is homogeneous.
        with_lang = [u for u in updates if "lang" in u]
        without_lang = [u for u in updates if "lang" not in u]
        for group in (with_lang, without_lang):
            if group:
                db.execute(update(Sample), group)
        db.commit()


def _record_error(sample_ids: list[int], msg: str, dataset_id: int | None = None) -> None:
    with SessionLocal() as db:
        if dataset_id is not None:
            sample_ids = list(db.scalars(select(Sample.id).where(Sample.dataset_id == dataset_id)))
        existing = set(db.scalars(select(Sample.id).where(Sample.id.in_(sample_ids)))) if sample_ids else set()
        store_results(db, {sid: [_error_result(msg)] for sid in sample_ids if sid in existing}, dataset_id=dataset_id)


def qc_dataset_sync(dataset_id: int) -> dict[str, Any] | None:
    t0 = time.perf_counter()
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            log.warning("qc: dataset %s not found", dataset_id)
            return None
        rows = load_rows(db, dataset_id)
    t_load = time.perf_counter()
    stats = compute_stats(rows)
    t_stats = time.perf_counter()
    results = evaluate(rows, stats)
    t_checks = time.perf_counter()
    with SessionLocal() as db:
        # Samples deleted while we were working are skipped.
        alive = set(db.scalars(select(Sample.id).where(Sample.dataset_id == dataset_id)))
        results = {sid: res for sid, res in results.items() if sid in alive}
        store_results(db, results, dataset_id=dataset_id)
    t_store = time.perf_counter()

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for res in results.values():
        status, _ = summarize(res)
        counts[status] = counts.get(status, 0) + 1
    log.info(
        "qc: dataset %s, %d samples -> %s (load %.1fs, stats %.1fs, checks %.1fs, store %.1fs)",
        dataset_id,
        len(results),
        counts,
        t_load - t0,
        t_stats - t_load,
        t_checks - t_stats,
        t_store - t_checks,
    )
    return {"dataset_id": dataset_id, "samples": len(results), "counts": counts, "seconds": round(t_store - t0, 2)}


def qc_sample_sync(sample_id: int) -> dict[str, Any] | None:
    with SessionLocal() as db:
        sample = db.get(Sample, sample_id)
        if sample is None:
            log.warning("qc: sample %s not found", sample_id)
            return None
        rows = load_rows(db, sample.dataset_id)
    stats = compute_stats(rows)  # dataset-wide inputs (duplicates, length distribution)
    row = next((r for r in rows if r.id == sample_id), None)
    if row is None:
        return None
    results = evaluate([row], stats)
    with SessionLocal() as db:
        if db.get(Sample, sample_id) is None:
            return None
        store_results(db, results)
    status, score = summarize(results[sample_id])
    log.info("qc: sample %s -> %s (score %s)", sample_id, status, score)
    return {"sample_id": sample_id, "qc_status": status, "qc_score": score}


async def run_dataset_qc(ctx: dict[str, Any] | None, dataset_id: int) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(qc_dataset_sync, dataset_id)
    except Exception as exc:
        log.exception("qc: dataset %s failed", dataset_id)
        msg = _exc_message(exc)
        try:
            await asyncio.to_thread(_record_error, [], msg, dataset_id)
        except Exception:
            log.exception("qc: could not record qc_error for dataset %s", dataset_id)
        return {"dataset_id": dataset_id, "error": msg}


async def run_sample_qc(ctx: dict[str, Any] | None, sample_id: int) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(qc_sample_sync, sample_id)
    except Exception as exc:
        log.exception("qc: sample %s failed", sample_id)
        msg = _exc_message(exc)
        try:
            await asyncio.to_thread(_record_error, [sample_id], msg)
        except Exception:
            log.exception("qc: could not record qc_error for sample %s", sample_id)
        return {"sample_id": sample_id, "qc_status": "fail", "error": msg}

"""Query helpers shared by the samples list, search, semantic search and export."""

from collections.abc import Sequence

from sqlalchemy import Select, exists, func, or_, select, text
from sqlalchemy.orm import Session

from app.models import Embedding, QCResult, Sample
from app.schemas import SampleFilter, SampleOut, SimilarSample

PREVIEW_CHARS = 160


def apply_sample_filter(stmt: Select, f: SampleFilter) -> Select:
    if f.dataset_id is not None:
        stmt = stmt.where(Sample.dataset_id == f.dataset_id)
    if f.qc_status:
        stmt = stmt.where(Sample.qc_status.in_(f.qc_status))
    if f.category:
        stmt = stmt.where(Sample.category.ilike(f.category))
    if f.lang:
        stmt = stmt.where(Sample.lang == f.lang)
    if f.min_tokens is not None:
        stmt = stmt.where(Sample.tokens_est >= f.min_tokens)
    if f.max_tokens is not None:
        stmt = stmt.where(Sample.tokens_est <= f.max_tokens)
    if f.text_contains:
        pattern = f"%{f.text_contains}%"
        stmt = stmt.where(or_(Sample.prompt.ilike(pattern), Sample.context.ilike(pattern), Sample.response.ilike(pattern)))
    if f.failed_check:
        stmt = stmt.where(
            exists().where(
                QCResult.sample_id == Sample.id,
                QCResult.check_name == f.failed_check,
                QCResult.severity.in_(["warn", "fail"]),
            )
        )
    return stmt


def find_samples(db: Session, f: SampleFilter, limit: int = 50, offset: int = 0) -> tuple[list[Sample], int]:
    base = apply_sample_filter(select(Sample), f)
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.scalars(base.order_by(Sample.dataset_id, Sample.sample_index).limit(limit).offset(offset)).all()
    return list(rows), total


def nearest_samples(
    db: Session, vector: Sequence[float], f: SampleFilter, model: str, limit: int, exclude_id: int | None = None
) -> list[SimilarSample]:
    """Top-`limit` samples matching `f`, closest to `vector` by cosine distance (pgvector `<=>`).

    `ORDER BY distance LIMIT k` is what lets Postgres walk the HNSW index instead of scanning every
    vector. Filters are applied while walking it: with pgvector >= 0.8 iterative scans the index keeps
    returning candidates until `limit` rows pass the WHERE clause (without it, a selective filter
    could leave fewer than k results out of the default ef_search = 40 candidates).
    """
    distance = Embedding.embedding.cosine_distance(vector).label("distance")
    stmt = select(Sample, distance).join(Embedding, Embedding.sample_id == Sample.id).where(Embedding.model == model)
    stmt = apply_sample_filter(stmt, f)
    if exclude_id is not None:
        stmt = stmt.where(Sample.id != exclude_id)
    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))  # this transaction only
    rows = db.execute(stmt.order_by(distance).limit(limit)).all()
    return [SimilarSample(**sample_out(s).model_dump(), similarity=round(1 - float(d), 4)) for s, d in rows]


def sample_out(s: Sample) -> SampleOut:
    preview = " ".join((s.prompt or "").split())
    if len(preview) > PREVIEW_CHARS:
        preview = preview[: PREVIEW_CHARS - 1] + "…"
    return SampleOut(
        id=s.id,
        dataset_id=s.dataset_id,
        sample_index=s.sample_index,
        category=s.category,
        prompt_preview=preview,
        tokens_est=s.tokens_est,
        lang=s.lang,
        qc_status=s.qc_status,
        qc_score=s.qc_score,
    )

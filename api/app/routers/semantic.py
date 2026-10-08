"""Semantic search over sample embeddings (see docs/api-contract.md, "Semantic search").

The query text is embedded in the API process (a few ms on CPU once the model is loaded) and
matched against the stored sample vectors with pgvector; the samples themselves are embedded by the
`embed_dataset` worker job.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import embeddings, queue
from app.db import get_db
from app.models import EMBEDDING_DIM, Dataset, Embedding, Sample
from app.queries import nearest_samples
from app.schemas import EmbeddingStatus, Enqueued, SampleFilter, SemanticSearchRequest, SemanticSearchResponse, SimilarSample

log = logging.getLogger(__name__)

router = APIRouter(tags=["semantic"])


def _dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    return ds


def embedding_status(db: Session, dataset_id: int) -> EmbeddingStatus:
    total = db.scalar(select(func.count()).where(Sample.dataset_id == dataset_id)) or 0
    embedded = (
        db.scalar(
            select(func.count())
            .select_from(Embedding)
            .join(Sample, Sample.id == Embedding.sample_id)
            .where(Sample.dataset_id == dataset_id, Embedding.model == embeddings.MODEL_NAME)
        )
        or 0
    )
    return EmbeddingStatus(model=embeddings.MODEL_NAME, dim=EMBEDDING_DIM, embedded=embedded, total=total)


@router.get("/datasets/{dataset_id}/embeddings", response_model=EmbeddingStatus)
def get_embedding_status(dataset_id: int, db: Session = Depends(get_db)) -> EmbeddingStatus:
    _dataset_or_404(db, dataset_id)
    return embedding_status(db, dataset_id)


@router.post("/datasets/{dataset_id}/embeddings", status_code=202, response_model=Enqueued)
async def embed_dataset(dataset_id: int, db: Session = Depends(get_db)) -> Enqueued:
    """(Re-)embed the dataset's samples that have no vector yet. Also runs automatically after import."""
    _dataset_or_404(db, dataset_id)
    try:
        await queue.enqueue("embed_dataset", dataset_id)
    except Exception as exc:
        log.exception("semantic: enqueue embed_dataset(%s) failed", dataset_id)
        raise HTTPException(status_code=503, detail="job queue unavailable") from exc
    return Enqueued(enqueued=1)


def _embed_query(query: str) -> list[float]:
    try:
        return embeddings.embed_query(query)
    except Exception as exc:  # model download / load failure
        log.exception("semantic: could not embed query")
        raise HTTPException(status_code=503, detail="embedding model unavailable") from exc


@router.post("/search/semantic", response_model=SemanticSearchResponse)
def semantic_search(body: SemanticSearchRequest, db: Session = Depends(get_db)) -> SemanticSearchResponse:
    """Samples whose meaning is closest to the query, optionally narrowed by a SampleFilter."""
    _dataset_or_404(db, body.dataset_id)
    if embedding_status(db, body.dataset_id).embedded == 0:
        raise HTTPException(status_code=409, detail="this dataset has no embeddings yet")
    vector = _embed_query(body.query)
    f = body.filter.model_copy(update={"dataset_id": body.dataset_id})
    items = nearest_samples(db, vector, f, embeddings.MODEL_NAME, body.limit)
    return SemanticSearchResponse(model=embeddings.MODEL_NAME, items=items)


@router.get("/samples/{sample_id}/similar", response_model=list[SimilarSample])
def similar_samples(
    sample_id: int, limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)
) -> list[SimilarSample]:
    """Nearest neighbours of a sample within its own dataset (the sample itself is excluded)."""
    s = db.get(Sample, sample_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} not found")
    vector = db.scalar(
        select(Embedding.embedding).where(Embedding.sample_id == sample_id, Embedding.model == embeddings.MODEL_NAME)
    )
    if vector is None:
        raise HTTPException(status_code=409, detail=f"sample {sample_id} has no embedding yet")
    f = SampleFilter(dataset_id=s.dataset_id)
    return nearest_samples(db, vector, f, embeddings.MODEL_NAME, limit, exclude_id=sample_id)

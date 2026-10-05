from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import nlsearch, queries
from app.db import get_db
from app.models import Dataset, Sample
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """Natural-language sample search. The query becomes a validated SampleFilter, never SQL."""
    if body.dataset_id is not None and db.get(Dataset, body.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"dataset {body.dataset_id} not found")
    f, parser = nlsearch.parse(body.query, body.dataset_id)
    if f.category:
        # canonical names ("brainstorming") -> the dataset's own spelling ("Brainstorm")
        stmt = select(Sample.category).distinct().where(Sample.category.is_not(None))
        if f.dataset_id is not None:
            stmt = stmt.where(Sample.dataset_id == f.dataset_id)
        f.category = nlsearch.resolve_category(f.category, db.scalars(stmt.limit(500)).all())
    rows, total = queries.find_samples(db, f, limit=body.limit)
    return SearchResponse(filter=f, parser=parser, items=[queries.sample_out(s) for s in rows], total=total)

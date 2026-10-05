from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import storage
from app.db import get_db
from app.models import Dataset, Export
from app.queue import enqueue
from app.schemas import ExportCreate, ExportOut

router = APIRouter(tags=["exports"])


def _get_export(db: Session, export_id: int) -> Export:
    export = db.get(Export, export_id)
    if export is None:
        raise HTTPException(status_code=404, detail=f"export {export_id} not found")
    return export


def export_filename(hf_repo_id: str, export_id: int) -> str:
    repo_name = hf_repo_id.rsplit("/", 1)[-1] or "dataset"
    repo_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in repo_name)
    return f"{repo_name}-export-{export_id}.zip"


@router.post("/exports", status_code=202, response_model=ExportOut)
async def create_export(body: ExportCreate, db: Session = Depends(get_db)) -> Export:
    if db.get(Dataset, body.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"dataset {body.dataset_id} not found")
    export = Export(
        dataset_id=body.dataset_id,
        filter=body.filter.model_dump(exclude_none=True, exclude={"dataset_id"}),
        val_ratio=body.val_ratio,
        format=body.format,
        status="pending",
        num_samples=0,
    )
    db.add(export)
    db.commit()
    db.refresh(export)
    try:
        await enqueue("build_export", export.id)
    except Exception as exc:
        export.status = "failed"
        export.error = "could not enqueue export job"
        db.commit()
        raise HTTPException(status_code=503, detail="job queue unavailable") from exc
    return export


@router.get("/exports", response_model=list[ExportOut])
def list_exports(dataset_id: int | None = Query(default=None), db: Session = Depends(get_db)) -> list[Export]:
    stmt = select(Export).order_by(Export.created_at.desc(), Export.id.desc())
    if dataset_id is not None:
        stmt = stmt.where(Export.dataset_id == dataset_id)
    return list(db.scalars(stmt).all())


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: int, db: Session = Depends(get_db)) -> Export:
    return _get_export(db, export_id)


@router.get("/exports/{export_id}/download")
def download_export(export_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    export = _get_export(db, export_id)
    if export.status != "ready" or not export.s3_key:
        raise HTTPException(status_code=409, detail=f"export is not ready (status: {export.status})")
    dataset = db.get(Dataset, export.dataset_id)
    filename = export_filename(dataset.hf_repo_id if dataset else "dataset", export.id)
    try:
        meta = storage.head(export.s3_key)
        _, chunks = storage.stream(export.s3_key)
    except ClientError as exc:
        raise HTTPException(status_code=404, detail="export archive is missing from storage") from exc
    return StreamingResponse(
        chunks,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(meta["ContentLength"]),
        },
    )

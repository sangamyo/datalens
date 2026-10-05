from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import datasets

app = FastAPI(title="EpisodeHub API", version="0.1.0")
app.include_router(datasets.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Deliberately touches no dependencies."""
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, str]:
    """Readiness: we can actually reach Postgres."""
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ready"}

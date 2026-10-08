"""Demo entrypoint: the API under /api (as the Vite dev proxy does) and the built UI at /."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.main import app as api

DIST = Path(__file__).parent / "web-dist"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", api)
app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str) -> FileResponse:
    """Serve real files from the build, and index.html for client-side routes."""
    f = (DIST / path).resolve()
    if path and f.is_file() and DIST.resolve() in f.parents:
        return FileResponse(f)
    return FileResponse(DIST / "index.html")

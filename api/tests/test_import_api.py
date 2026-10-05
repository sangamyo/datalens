"""POST /datasets/import against SQLite with the job queue and Hub lookups stubbed (no Docker, no network)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Dataset
from app.routers import datasets as datasets_router


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Dataset.metadata.create_all(engine, tables=[Dataset.__table__])
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    calls: list[tuple] = []

    async def fake_enqueue(name, *args):
        calls.append((name, *args))

    monkeypatch.setattr(datasets_router, "enqueue", fake_enqueue)
    monkeypatch.setattr(datasets_router, "resolve_default_config", lambda repo: "main" if repo == "openai/gsm8k" else "default")

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app), calls
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_import_enqueues_with_fields_and_409(env) -> None:
    client, calls = env
    r = client.post("/datasets/import", json={"hf_repo_id": "openai/gsm8k", "max_samples": 300})
    assert r.status_code == 202
    body = r.json()
    assert (body["status"], body["config"], body["split"], body["name"]) == ("pending", "main", "train", "gsm8k")
    assert calls == [("import_dataset", body["id"], 300, None)]

    # Same repo, default config resolved to the same name -> 409, explicit config too.
    assert client.post("/datasets/import", json={"hf_repo_id": "openai/gsm8k"}).status_code == 409
    assert client.post("/datasets/import", json={"hf_repo_id": "openai/gsm8k", "config": "main"}).status_code == 409
    # Different split / config is a different dataset.
    assert client.post("/datasets/import", json={"hf_repo_id": "openai/gsm8k", "split": "test"}).status_code == 202
    r = client.post(
        "/datasets/import",
        json={"hf_repo_id": "openai/gsm8k", "config": "socratic", "fields": {"prompt": "question", "response": "answer"}},
    )
    assert r.status_code == 202
    assert calls[-1] == ("import_dataset", r.json()["id"], 2000, {"prompt": "question", "response": "answer"})


def test_import_validation(env) -> None:
    client, _ = env
    assert client.post("/datasets/import", json={"hf_repo_id": "x/y", "max_samples": 0}).status_code == 422
    assert client.post("/datasets/import", json={"hf_repo_id": "no-slash"}).status_code == 422

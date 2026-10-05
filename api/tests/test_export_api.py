"""Export + search routers against an in-memory SQLite DB (queue and storage mocked)."""

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.db import Base, get_db
from app.models import Dataset, Export, Sample
from app.routers import exports as exports_router
from app.routers import search as search_router
from app.routers.exports import export_filename

# Only the routers under test (the full app.main imports every router).
app = FastAPI()
app.include_router(exports_router.router)
app.include_router(search_router.router)


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    enqueued = []

    async def fake_enqueue(fn, *args):
        enqueued.append((fn, *args))

    monkeypatch.setattr(exports_router, "enqueue", fake_enqueue)
    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", None)
    monkeypatch.setattr(s, "anthropic_api_key", None)
    app.dependency_overrides[get_db] = override
    with Session() as db:
        ds = Dataset(hf_repo_id="HuggingFaceH4/no_robots", name="no_robots", status="ready")
        db.add(ds)
        db.flush()
        cats = ["Brainstorm", "Open QA", "Chat"]
        db.add_all(
            Sample(dataset_id=ds.id, sample_index=i, prompt=f"Prompt {i} about OpenAI" if i % 5 == 0 else f"Prompt {i}",
                   response="r", category=cats[i % 3], tokens_est=i * 10, qc_status="pass" if i % 2 else "fail")
            for i in range(30)
        )
        db.commit()
        ds_id = ds.id
    yield TestClient(app), Session, ds_id, enqueued
    app.dependency_overrides.clear()


def test_export_filename():
    assert export_filename("databricks/databricks-dolly-15k", 7) == "databricks-dolly-15k-export-7.zip"
    assert export_filename("a/b c", 1) == "b_c-export-1.zip"


def test_create_export(env):
    client, Session, ds_id, enqueued = env
    r = client.post("/exports", json={"dataset_id": ds_id, "filter": {"qc_status": ["pass"], "dataset_id": 99},
                                      "val_ratio": 0.2, "format": "chat"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending" and body["format"] == "chat" and body["val_ratio"] == 0.2
    assert body["filter"] == {"qc_status": ["pass"]}  # dataset_id is never stored in the filter
    assert enqueued == [("build_export", body["id"])]


def test_create_export_validation(env):
    client, _, ds_id, _ = env
    assert client.post("/exports", json={"dataset_id": 999}).status_code == 404
    assert client.post("/exports", json={"dataset_id": ds_id, "format": "csv"}).status_code == 422
    assert client.post("/exports", json={"dataset_id": ds_id, "val_ratio": 0.9}).status_code == 422


def test_create_export_queue_down(env, monkeypatch):
    client, Session, ds_id, _ = env

    async def down(*a):
        raise ConnectionError("redis down")

    monkeypatch.setattr(exports_router, "enqueue", down)
    r = client.post("/exports", json={"dataset_id": ds_id})
    assert r.status_code == 503
    with Session() as db:
        e = db.query(Export).one()
        assert e.status == "failed"


def test_list_get_and_download(env, monkeypatch):
    client, Session, ds_id, _ = env
    ids = [client.post("/exports", json={"dataset_id": ds_id}).json()["id"] for _ in range(3)]
    listed = client.get("/exports", params={"dataset_id": ds_id}).json()
    assert [e["id"] for e in listed] == sorted(ids, reverse=True)
    assert client.get("/exports", params={"dataset_id": 999}).json() == []
    assert client.get(f"/exports/{ids[0]}").json()["id"] == ids[0]
    assert client.get("/exports/999").status_code == 404

    r = client.get(f"/exports/{ids[0]}/download")
    assert r.status_code == 409

    with Session() as db:
        e = db.get(Export, ids[0])
        e.status, e.s3_key, e.num_samples = "ready", f"exports/{ids[0]}.zip", 3
        db.commit()
    payload = b"PK\x03\x04fakezip"
    monkeypatch.setattr(exports_router.storage, "head", lambda key: {"ContentLength": len(payload)})
    monkeypatch.setattr(exports_router.storage, "stream", lambda key: ({}, iter([payload[:4], payload[4:]])))
    r = client.get(f"/exports/{ids[0]}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="no_robots-export-{ids[0]}.zip"'
    assert r.headers["content-length"] == str(len(payload))
    assert r.content == payload


def test_download_missing_object(env, monkeypatch):
    client, Session, ds_id, _ = env
    eid = client.post("/exports", json={"dataset_id": ds_id}).json()["id"]
    with Session() as db:
        e = db.get(Export, eid)
        e.status, e.s3_key = "ready", "exports/x.zip"
        db.commit()

    def missing(key):
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    monkeypatch.setattr(exports_router.storage, "head", missing)
    assert client.get(f"/exports/{eid}/download").status_code == 404


def test_search_endpoint(env):
    client, _, ds_id, _ = env
    r = client.post("/search", json={"query": "failed brainstorming samples", "dataset_id": ds_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parser"] == "rules"
    # canonical "brainstorming" resolved to the dataset's own "Brainstorm"
    assert body["filter"]["category"] == "Brainstorm" and body["filter"]["qc_status"] == ["fail"]
    assert body["total"] == 5
    assert all(i["category"] == "Brainstorm" and i["qc_status"] == "fail" for i in body["items"])

    r = client.post("/search", json={"query": 'samples mentioning "openai" over 100 tokens', "dataset_id": ds_id,
                                     "limit": 2})
    body = r.json()
    assert body["filter"]["text_contains"] == "openai" and body["filter"]["min_tokens"] == 101
    assert body["total"] == 3 and len(body["items"]) == 2  # samples 15, 20, 25
    assert set(body["items"][0]) >= {"id", "prompt_preview", "tokens_est", "qc_status"}

    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "refusals", "dataset_id": 10**9}).status_code == 404

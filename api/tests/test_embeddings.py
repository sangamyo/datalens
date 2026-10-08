"""Embedding job and semantic-search endpoints against in-memory SQLite (no Docker, no model download).

The model is replaced by tests/fake_embed.py. Vector *ranking* needs pgvector's `<=>` operator, so it
is tested against real Postgres in test_semantic_pg.py; here we cover the job, status, enqueueing and
every error path, which don't depend on the distance operator.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import embeddings, queue
from app.db import Base, get_db
from app.jobs import embed_job, import_job
from app.jobs.import_job import build_sample
from app.models import EMBEDDING_DIM, Dataset, Embedding, Sample
from app.routers import semantic as semantic_router
from tests.fake_embed import FakeEmbedder, fake_vector

ROWS = [
    ("What is the capital of France?", None, "Paris is the capital of France."),
    ("Write a short poem about the sea.", None, "Waves roll in, waves roll out."),
    ("How do I bake sourdough bread?", "Some context that is not embedded.", "Mix flour, water and starter."),
    ("Explain photosynthesis.", None, "Plants turn light, water and CO2 into sugar."),
    ("Name three primary colours.", None, "Red, yellow and blue."),
]


@pytest.fixture()
def fake_model(monkeypatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(embeddings, "embed_texts", fake)
    return fake


@pytest.fixture()
def session_factory(monkeypatch, fake_model):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[Dataset.__table__, Sample.__table__, Embedding.__table__])
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(embed_job, "SessionLocal", factory)
    with factory() as db:
        db.add_all(
            [
                Dataset(id=1, hf_repo_id="org/a", config="default", split="train", name="a", status="ready"),
                Dataset(id=2, hf_repo_id="org/b", config="default", split="train", name="b", status="ready"),
            ]
        )
        db.flush()
        db.add_all(Sample(**build_sample(1, i, p, c, r, None)) for i, (p, c, r) in enumerate(ROWS))
        db.add(Sample(**build_sample(2, 0, "other dataset", None, "x", None)))
        db.commit()
    yield factory
    engine.dispose()


def _vectors(factory, dataset_id: int = 1) -> dict[int, tuple[str, list[float]]]:
    with factory() as db:
        rows = db.execute(
            select(Embedding.sample_id, Embedding.model, Embedding.embedding)
            .join(Sample, Sample.id == Embedding.sample_id)
            .where(Sample.dataset_id == dataset_id)
        ).all()
    return {sid: (model, list(vec)) for sid, model, vec in rows}


# ---------------------------------------------------------------- text + query helpers
def test_sample_text_is_prompt_and_response_truncated() -> None:
    assert embeddings.sample_text("  Q?  ", " A. ") == "Q?\n\nA."
    assert embeddings.sample_text(None, "only answer") == "only answer"
    assert embeddings.sample_text("", "") == ""
    assert len(embeddings.sample_text("x" * 5000, "y")) == embeddings.MAX_CHARS


def test_embed_query_uses_bge_query_instruction(fake_model) -> None:
    v = embeddings.embed_query("  bread recipes ")
    assert fake_model.batches == [[embeddings.QUERY_PREFIX + "bread recipes"]]
    assert len(v) == EMBEDDING_DIM


def test_fake_vectors_are_normalised_and_deterministic() -> None:
    v = fake_vector("hello world")
    assert v == fake_vector("Hello, WORLD!")
    assert abs(sum(x * x for x in v) - 1) < 1e-9


# ---------------------------------------------------------------- job
def test_embed_job_embeds_every_sample_in_batches(session_factory, fake_model, monkeypatch) -> None:
    monkeypatch.setattr(embed_job, "WRITE_BATCH", 2)
    out = asyncio.run(embed_job.embed_dataset({}, 1))
    assert out["dataset_id"] == 1 and out["embedded"] == len(ROWS)
    assert [len(b) for b in fake_model.batches] == [2, 2, 1]
    sent = [t for b in fake_model.batches for t in b]
    assert "How do I bake sourdough bread?\n\nMix flour, water and starter." in sent  # context left out
    assert [len(t) for t in sent] == sorted(len(t) for t in sent)  # shortest first: minimal padding

    vecs = _vectors(session_factory)
    assert len(vecs) == len(ROWS)
    for model, vec in vecs.values():
        assert model == embeddings.MODEL_NAME and len(vec) == EMBEDDING_DIM
    assert _vectors(session_factory, dataset_id=2) == {}  # other datasets untouched


def test_embed_job_is_incremental_and_replaces_other_models(session_factory, fake_model) -> None:
    asyncio.run(embed_job.embed_dataset({}, 1))
    before = _vectors(session_factory)
    fake_model.batches.clear()

    again = asyncio.run(embed_job.embed_dataset({}, 1))
    assert again["embedded"] == 0 and fake_model.batches == []
    assert _vectors(session_factory) == before

    # one stale vector from an older model + one missing vector -> both (and only they) are re-embedded
    sids = sorted(before)
    with session_factory() as db:
        db.query(Embedding).filter(Embedding.sample_id == sids[0]).update({"model": "old-model"})
        db.query(Embedding).filter(Embedding.sample_id == sids[1]).delete()
        db.commit()
    out = asyncio.run(embed_job.embed_dataset({}, 1))
    assert out["embedded"] == 2
    vecs = _vectors(session_factory)
    assert len(vecs) == len(ROWS) and {m for m, _ in vecs.values()} == {embeddings.MODEL_NAME}


def test_embed_job_missing_dataset_and_model_failure(session_factory, monkeypatch) -> None:
    assert asyncio.run(embed_job.embed_dataset({}, 99)) is None

    monkeypatch.setattr(embed_job, "WRITE_BATCH", 2)
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("onnx exploded")
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(embeddings, "embed_texts", flaky)
    out = asyncio.run(embed_job.embed_dataset({}, 1))
    assert out["error"].startswith("RuntimeError: onnx exploded")
    assert len(_vectors(session_factory)) == 2  # the first committed batch is kept
    monkeypatch.setattr(embeddings, "embed_texts", FakeEmbedder())
    assert asyncio.run(embed_job.embed_dataset({}, 1))["embedded"] == len(ROWS) - 2  # resumes


def test_import_job_enqueues_qc_and_embeddings(monkeypatch) -> None:
    enqueued: list[tuple] = []

    class FakeRedis:
        async def enqueue_job(self, name, *args):
            enqueued.append((name, *args))

    monkeypatch.setattr(import_job, "_import_sync", lambda *a: 3)
    assert asyncio.run(import_job.import_dataset({"redis": FakeRedis()}, 7, 10)) == 3
    assert enqueued == [("run_dataset_qc", 7), ("embed_dataset", 7)]


# ---------------------------------------------------------------- router (non-ranking paths)
@pytest.fixture()
def client(session_factory, monkeypatch):
    calls: list[tuple] = []

    async def fake_enqueue(function, *args):
        calls.append((function, *args))

    monkeypatch.setattr(queue, "enqueue", fake_enqueue)
    app = FastAPI()
    app.include_router(semantic_router.router)

    def db_override():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = db_override
    c = TestClient(app)
    c.calls = calls
    return c


def test_status_and_enqueue(client, session_factory) -> None:
    st = client.get("/datasets/1/embeddings").json()
    assert st == {"model": embeddings.MODEL_NAME, "dim": EMBEDDING_DIM, "embedded": 0, "total": len(ROWS)}

    r = client.post("/datasets/1/embeddings")
    assert r.status_code == 202 and r.json() == {"enqueued": 1}
    assert client.calls == [("embed_dataset", 1)]

    asyncio.run(embed_job.embed_dataset({}, 1))
    assert client.get("/datasets/1/embeddings").json()["embedded"] == len(ROWS)
    assert client.get("/datasets/2/embeddings").json()["embedded"] == 0

    assert client.get("/datasets/99/embeddings").status_code == 404
    assert client.post("/datasets/99/embeddings").status_code == 404


def test_enqueue_queue_down_is_503(client, monkeypatch) -> None:
    async def down(*args):
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue, "enqueue", down)
    r = client.post("/datasets/1/embeddings")
    assert r.status_code == 503 and r.json()["detail"] == "job queue unavailable"


def test_semantic_search_errors(client, monkeypatch) -> None:
    assert client.post("/search/semantic", json={"query": "x", "dataset_id": 99}).status_code == 404
    r = client.post("/search/semantic", json={"query": "bread", "dataset_id": 1})
    assert r.status_code == 409 and "no embeddings" in r.json()["detail"]
    assert client.post("/search/semantic", json={"query": "", "dataset_id": 1}).status_code == 422
    assert client.post("/search/semantic", json={"query": "x", "dataset_id": 1, "limit": 0}).status_code == 422
    assert client.post("/search/semantic", json={"query": "x"}).status_code == 422  # dataset_id required

    asyncio.run(embed_job.embed_dataset({}, 1))

    def broken(texts):
        raise OSError("model download failed")

    monkeypatch.setattr(embeddings, "embed_texts", broken)
    r = client.post("/search/semantic", json={"query": "bread", "dataset_id": 1})
    assert r.status_code == 503 and r.json()["detail"] == "embedding model unavailable"


def test_similar_errors(client, session_factory) -> None:
    assert client.get("/samples/999/similar").status_code == 404
    r = client.get("/samples/1/similar")
    assert r.status_code == 409 and "no embedding" in r.json()["detail"]
    assert client.get("/samples/1/similar", params={"limit": 51}).status_code == 422

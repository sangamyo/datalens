"""Semantic search ranking against real Postgres + pgvector (cosine distance, HNSW index).

Needs a throwaway database; skipped unless TEST_DATABASE_URL is set, e.g. with docker compose up:

    TEST_DATABASE_URL=postgresql+psycopg://datalens:<password>@localhost:5434/datalens_test

All tables in that database are dropped and recreated. The embedding model is the deterministic fake
from tests/fake_embed.py (bag-of-words hashing), so "closest" means "shares the most words".
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app import embeddings
from app.db import Base, get_db
from app.jobs import embed_job
from app.jobs.import_job import build_sample
from app.main import app
from app.models import Dataset, Sample
from tests.fake_embed import FakeEmbedder

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set (needs Postgres with pgvector)")

ROWS = [  # (prompt, response, category)
    ("How do I bake sourdough bread at home?", "Mix flour, water and a sourdough starter, then bake.", "open_qa"),
    ("Give me a recipe for sourdough bread.", "Flour, water, salt and starter; bake the sourdough at 230C.", "brainstorming"),
    ("What is the capital of France?", "Paris is the capital of France.", "open_qa"),
    ("Write a poem about the sea.", "Waves roll in and out under a grey sky.", "creative_writing"),
    ("Explain photosynthesis to a child.", "Plants use sunlight to turn water and air into food.", "open_qa"),
    ("Which planet is the largest?", "Jupiter is the largest planet in the solar system.", "open_qa"),
]


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(URL, pool_pre_ping=True)
    try:
        with eng.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except OperationalError as exc:
        pytest.skip(f"cannot reach TEST_DATABASE_URL: {exc}")
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)  # includes the HNSW index declared on Embedding
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def client(engine, monkeypatch):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE datasets RESTART IDENTITY CASCADE"))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(embed_job, "SessionLocal", factory)
    monkeypatch.setattr(embeddings, "embed_texts", FakeEmbedder())
    with factory() as db:
        db.add_all(
            [
                Dataset(id=1, hf_repo_id="org/a", config="default", split="train", name="a", status="ready"),
                Dataset(id=2, hf_repo_id="org/b", config="default", split="train", name="b", status="ready"),
            ]
        )
        db.flush()
        db.add_all(Sample(**build_sample(1, i, p, None, r, c)) for i, (p, r, c) in enumerate(ROWS))
        db.add(Sample(**build_sample(2, 0, "Sourdough bread sourdough bread", None, "sourdough bread", "open_qa")))
        db.commit()
    for dataset_id in (1, 2):
        asyncio.run(embed_job.embed_dataset({}, dataset_id))

    def override():
        with factory() as s:
            yield s

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _search(client, query: str, **extra) -> list[dict]:
    r = client.post("/search/semantic", json={"query": query, "dataset_id": 1, **extra})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == embeddings.MODEL_NAME
    return body["items"]


def test_semantic_search_ranks_by_cosine_similarity(client) -> None:
    items = _search(client, "sourdough bread")
    assert {it["sample_index"] for it in items[:2]} == {0, 1}  # the two bread samples first
    sims = [it["similarity"] for it in items]
    assert sims == sorted(sims, reverse=True)
    assert all(-1 <= s <= 1 for s in sims) and sims[0] > sims[-1]
    assert len(items) == len(ROWS) and {it["dataset_id"] for it in items} == {1}  # other dataset excluded
    assert items[0]["prompt_preview"] and "qc_status" in items[0]

    assert [it["sample_index"] for it in _search(client, "capital of France", limit=1)] == [2]


def test_semantic_search_combines_with_sample_filter(client) -> None:
    items = _search(client, "sourdough bread", filter={"category": "brainstorming"})
    assert [it["sample_index"] for it in items] == [1]
    items = _search(client, "sourdough bread", filter={"text_contains": "planet", "dataset_id": 2})
    assert [it["sample_index"] for it in items] == [5]  # filter's own dataset_id is ignored
    assert _search(client, "sourdough bread", filter={"qc_status": ["fail"]}) == []


def test_similar_samples(client) -> None:
    r = client.get("/samples/1/similar", params={"limit": 3})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 3
    assert items[0]["id"] == 2  # the other bread sample, not itself and not dataset 2's bread row
    assert all(it["id"] != 1 and it["dataset_id"] == 1 for it in items)
    sims = [it["similarity"] for it in items]
    assert sims == sorted(sims, reverse=True)


def test_hnsw_index_exists_and_is_used(client, engine) -> None:
    with engine.connect() as conn:
        indexdef = conn.scalar(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_embeddings_embedding_hnsw'")
        )
        assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef
        # With only a few rows the planner prefers a seq scan; forbid it to check the index is usable
        # for ORDER BY embedding <=> query LIMIT k.
        conn.execute(text("SET enable_seqscan = off"))
        vec = "[" + ",".join(["0.05"] * 384) + "]"
        plan = conn.execute(
            text("EXPLAIN SELECT id FROM embeddings ORDER BY embedding <=> CAST(:v AS vector) LIMIT 5"), {"v": vec}
        ).scalars().all()
    assert any("ix_embeddings_embedding_hnsw" in line for line in plan)

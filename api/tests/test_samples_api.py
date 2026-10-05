"""Samples / dataset-detail endpoints and helpers against an in-memory SQLite database (no Docker).

Only the tables these endpoints touch are created (the embeddings table needs pgvector)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.jobs.import_job import build_sample
from app.main import app
from app.models import Dataset, Export, QCResult, Sample
from app.routers.datasets import percentile_nearest_rank, token_histogram


# ---------------------------------------------------------------- histogram (pure)
def test_histogram_empty_and_constant() -> None:
    assert token_histogram([]) == []
    [b] = token_histogram([5, 5, 5])
    assert (b.start, b.end, b.count) == (5, 5, 3)


def test_histogram_equal_width_up_to_p99_with_open_last_bucket() -> None:
    values = list(range(100)) + [10_000]  # one huge outlier
    buckets = token_histogram(values)
    assert len(buckets) <= 12
    assert sum(b.count for b in buckets) == len(values)
    assert buckets[0].start == 0
    widths = {b.end - b.start for b in buckets[:-1]}
    assert widths == {9}  # ceil((p99=99 - 0) / 12)
    assert buckets[-1].end == 10_000  # open-ended: holds the outlier
    for prev, nxt in zip(buckets, buckets[1:]):
        assert prev.end == nxt.start


def test_histogram_small_range() -> None:
    buckets = token_histogram([1, 2, 3])
    assert [(b.start, b.end, b.count) for b in buckets] == [(1, 2, 1), (2, 3, 2)]


def test_percentile_nearest_rank() -> None:
    assert percentile_nearest_rank(list(range(1, 101)), 99) == 99
    assert percentile_nearest_rank([7], 99) == 7


# ---------------------------------------------------------------- API on SQLite
@pytest.fixture()
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    tables = [Dataset.__table__, Sample.__table__, QCResult.__table__, Export.__table__]
    Dataset.metadata.create_all(engine, tables=tables)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        db.add_all(
            [
                Dataset(id=1, hf_repo_id="org/a", config="default", split="train", name="a", status="ready", num_samples=4),
                Dataset(id=2, hf_repo_id="org/b", config="default", split="train", name="b", status="ready", num_samples=1),
            ]
        )
        db.flush()
        rows = [
            build_sample(1, 0, "What is the capital of France?", None, "Paris.", "open_qa"),
            build_sample(1, 2, "Summarise this", "long text " * 50, "Short.", "summarization"),
            build_sample(1, 5, "Write a poem", None, "Roses are red " * 20, "creative_writing"),
            build_sample(1, 9, "Another question about France", None, "Yes.", None),
            build_sample(2, 1, "other dataset", None, "x", "open_qa"),
        ]
        db.add_all(Sample(**r) for r in rows)
        db.flush()
        db.add_all(
            [
                QCResult(sample_id=1, check_name="pii", passed=False, severity="warn", message="email", details={"spans": []}),
                QCResult(sample_id=1, check_name="empty_or_short", passed=True, severity="pass", message="ok", details={}),
            ]
        )
        db.query(Sample).filter(Sample.id == 1).update({"qc_status": "warn"})
        db.commit()

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_list_samples_and_filters(client) -> None:
    r = client.get("/datasets/1/samples")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 4
    assert [s["sample_index"] for s in body["items"]] == [0, 2, 5, 9]
    assert body["items"][0]["prompt_preview"] == "What is the capital of France?"

    assert client.get("/datasets/1/samples", params={"category": "OPEN_QA"}).json()["total"] == 1
    assert client.get("/datasets/1/samples", params={"text_contains": "france"}).json()["total"] == 2
    big = client.get("/datasets/1/samples", params={"min_tokens": 50}).json()
    assert {s["sample_index"] for s in big["items"]} == {2, 5}
    assert client.get("/datasets/1/samples", params={"max_tokens": 20}).json()["total"] == 2
    assert client.get("/datasets/1/samples", params=[("qc_status", "warn"), ("qc_status", "fail")]).json()["total"] == 1
    assert client.get("/datasets/1/samples", params={"failed_check": "pii"}).json()["total"] == 1
    page = client.get("/datasets/1/samples", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 4 and [s["sample_index"] for s in page["items"]] == [5, 9]


def test_list_samples_validation_and_404(client) -> None:
    assert client.get("/datasets/99/samples").status_code == 404
    assert client.get("/datasets/1/samples", params={"qc_status": "bogus"}).status_code == 422
    assert client.get("/datasets/1/samples", params={"limit": 0}).status_code == 422


def test_sample_detail_prev_next(client) -> None:
    first = client.get("/samples/1").json()
    assert first["prev_id"] is None and first["next_id"] == 2
    assert first["prompt"] == "What is the capital of France?" and first["response"] == "Paris."
    assert [q["check_name"] for q in first["qc_results"]] == ["empty_or_short", "pii"]
    middle = client.get("/samples/2").json()
    assert (middle["prev_id"], middle["next_id"]) == (1, 3)
    assert middle["context"].startswith("long text")
    last = client.get("/samples/4").json()
    assert (last["prev_id"], last["next_id"]) == (3, None)  # never crosses into dataset 2
    assert client.get("/samples/5").json()["prev_id"] is None
    assert client.get("/samples/999").status_code == 404


def test_dataset_detail(client) -> None:
    d = client.get("/datasets/1").json()
    assert d["qc_counts"] == {"pending": 3, "pass": 0, "warn": 1, "fail": 0}
    cats = {c["name"]: c["count"] for c in d["categories"]}
    assert cats == {"open_qa": 1, "summarization": 1, "creative_writing": 1, "uncategorised": 1}
    assert sum(b["count"] for b in d["token_histogram"]) == 4
    assert client.get("/datasets/42").status_code == 404
    assert [x["id"] for x in client.get("/datasets").json()] == [2, 1]

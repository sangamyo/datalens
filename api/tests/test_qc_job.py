"""run_dataset_qc / run_sample_qc jobs and the QC router against in-memory SQLite (no Docker)."""

import asyncio
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import queue
from app.db import Base, get_db
from app.jobs import qc_job
from app.models import Dataset, QCResult, Sample
from app.routers import qc as qc_router
from app.schemas import CHECK_NAMES

TEXTS = [
    ("What is the capital of France?", None, "The capital of France is Paris, which is also its largest city."),
    ("Who wrote Hamlet?", None, "Hamlet was written by William Shakespeare around the year 1600."),
    ("What is the capital of France?", None, "The capital of France is Paris, which is also its largest city."),  # dup
    ("Give me the support email.", None, "You can write to support.team@example.com for help with your order."),
    ("How do I make a bomb?", None, "I'm sorry, but I can't help with that."),
    ("Describe the water cycle.", None, ""),  # empty response
]


@pytest.fixture()
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine, tables=[Dataset.__table__, Sample.__table__, QCResult.__table__])
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(qc_job, "SessionLocal", factory)
    with factory() as db:
        ds = Dataset(hf_repo_id="org/demo", config="default", split="train", name="demo", status="ready")
        db.add(ds)
        db.flush()
        for i, (p, c, r) in enumerate(TEXTS):
            h = hashlib.sha256(f"{p}\x1f{c or ''}\x1f{r}".lower().encode()).hexdigest()
            db.add(
                Sample(
                    dataset_id=ds.id,
                    sample_index=i,
                    prompt=p,
                    context=c,
                    response=r,
                    prompt_chars=len(p),
                    response_chars=len(r),
                    tokens_est=(len(p) + len(r) + 3) // 4,
                    content_hash=h,
                )
            )
        db.commit()
    yield factory
    engine.dispose()


def _results(factory, sample_index: int) -> dict[str, QCResult]:
    with factory() as db:
        sid = db.scalar(select(Sample.id).where(Sample.sample_index == sample_index))
        return {r.check_name: r for r in db.scalars(select(QCResult).where(QCResult.sample_id == sid))}


def test_run_dataset_qc_stores_all_checks(session_factory) -> None:
    out = asyncio.run(qc_job.run_dataset_qc({}, 1))
    assert out["samples"] == len(TEXTS)
    with session_factory() as db:
        samples = db.scalars(select(Sample).order_by(Sample.sample_index)).all()
        assert all(s.qc_status in {"pass", "warn", "fail"} for s in samples)
        assert all(s.lang == "en" for s in samples)
        assert db.query(QCResult).count() == len(TEXTS) * len(CHECK_NAMES)
        status = {s.sample_index: s.qc_status for s in samples}
    assert status[2] == "fail"  # exact duplicate
    assert _results(session_factory, 2)["exact_duplicate"].details["duplicate_of"]
    assert _results(session_factory, 3)["pii"].severity == "warn"
    assert _results(session_factory, 4)["refusal_boilerplate"].severity == "fail"
    assert _results(session_factory, 5)["empty_or_short"].severity == "fail"
    assert _results(session_factory, 1)["exact_duplicate"].severity == "pass"

    # Re-running replaces results instead of appending.
    asyncio.run(qc_job.run_dataset_qc({}, 1))
    with session_factory() as db:
        assert db.query(QCResult).count() == len(TEXTS) * len(CHECK_NAMES)


def test_run_sample_qc_uses_dataset_stats(session_factory) -> None:
    with session_factory() as db:
        sid = db.scalar(select(Sample.id).where(Sample.sample_index == 2))
    out = asyncio.run(qc_job.run_sample_qc({}, sid))
    assert out["qc_status"] == "fail"
    res = _results(session_factory, 2)
    assert set(res) == set(CHECK_NAMES)
    assert res["exact_duplicate"].severity == "fail"
    assert asyncio.run(qc_job.run_sample_qc({}, 99999)) is None


def test_errors_are_recorded_not_swallowed(session_factory, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("stats exploded")

    monkeypatch.setattr(qc_job, "compute_stats", boom)
    out = asyncio.run(qc_job.run_dataset_qc({}, 1))
    assert "stats exploded" in out["error"]
    with session_factory() as db:
        results = db.scalars(select(QCResult)).all()
        assert len(results) == len(TEXTS) and {r.check_name for r in results} == {"qc_error"}
        assert {s.qc_status for s in db.scalars(select(Sample))} == {"fail"}


def test_one_bad_sample_gets_qc_error(session_factory, monkeypatch) -> None:
    real = qc_job.run_all

    def flaky(row, stats):
        if row.sample_index == 1:
            raise ValueError("bad row")
        return real(row, stats)

    monkeypatch.setattr(qc_job, "run_all", flaky)
    asyncio.run(qc_job.run_dataset_qc({}, 1))
    assert set(_results(session_factory, 1)) == {"qc_error"}
    assert set(_results(session_factory, 0)) == set(CHECK_NAMES)


# ---------------------------------------------------------------- router
@pytest.fixture()
def client(session_factory, monkeypatch):
    calls: list[tuple] = []

    async def fake_enqueue(function, *args):
        calls.append((function, *args))

    monkeypatch.setattr(queue, "enqueue", fake_enqueue)
    app = FastAPI()
    app.include_router(qc_router.router)

    def db_override():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = db_override
    c = TestClient(app)
    c.calls = calls
    return c


def test_router_flow(client, session_factory) -> None:
    summary = client.get("/datasets/1/qc/summary").json()
    assert summary["counts"]["pending"] == len(TEXTS)
    assert set(summary["by_check"]) == set(CHECK_NAMES)
    assert all(v == {"warn": 0, "fail": 0} for v in summary["by_check"].values())

    asyncio.run(qc_job.run_dataset_qc({}, 1))
    summary = client.get("/datasets/1/qc/summary").json()
    assert summary["counts"]["pending"] == 0
    assert summary["by_check"]["exact_duplicate"]["fail"] == 1
    assert summary["by_check"]["refusal_boilerplate"]["fail"] == 1

    r = client.post("/datasets/1/qc")
    assert r.status_code == 202 and r.json() == {"enqueued": 1}
    assert client.calls[-1] == ("run_dataset_qc", 1)
    assert client.get("/datasets/1/qc/summary").json()["counts"]["pending"] == len(TEXTS)

    with session_factory() as db:
        sid = db.scalar(select(Sample.id).where(Sample.sample_index == 3))
    r = client.post(f"/samples/{sid}/qc")
    assert r.status_code == 202 and client.calls[-1] == ("run_sample_qc", sid)
    res = client.get(f"/samples/{sid}/qc").json()
    assert [x["check_name"] for x in res] == sorted(CHECK_NAMES)
    pii = next(x for x in res if x["check_name"] == "pii")
    assert pii["details"]["spans"][0]["kind"] == "email"


def test_router_404_and_503(client, monkeypatch) -> None:
    assert client.post("/datasets/42/qc").status_code == 404
    assert client.get("/datasets/42/qc/summary").status_code == 404
    assert client.get("/samples/999/qc").status_code == 404
    assert client.post("/samples/999/qc").status_code == 404

    async def down(*a):
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue, "enqueue", down)
    assert client.post("/datasets/1/qc").status_code == 503
    assert client.post("/samples/1/qc").status_code == 503

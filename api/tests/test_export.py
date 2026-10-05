"""Export: pure helpers (split, formatting, manifest, zip) and the job end-to-end on SQLite."""

import asyncio
import json
import shutil
import time
import zipfile
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.jobs import export_job
from app.jobs.export_job import (
    build_manifest,
    format_line,
    format_record,
    split_ids,
    val_count,
    write_export_zip,
)
from app.models import Dataset, Export, QCResult, Sample


# ---------- pure helpers ----------
@pytest.mark.parametrize(
    "n,ratio,expected",
    [(0, 0.1, 0), (1, 0.5, 0), (1, 0.1, 0), (2, 0.1, 1), (2, 0.5, 1), (10, 0.1, 1), (10, 0.2, 2), (10, 0.0, 0),
     (20, 0.1, 2), (19, 0.1, 2), (3, 0.5, 2), (100, 0.25, 25), (5, 0.01, 1), (20000, 0.1, 2000)],
)
def test_val_count(n, ratio, expected):
    assert val_count(n, ratio) == expected


def test_split_is_deterministic_and_order_independent():
    ids = list(range(100, 150))
    a = split_ids(ids, 0.2, seed=7)
    b = split_ids(list(reversed(ids)), 0.2, seed=7)
    assert a == b
    train, val = a
    assert len(val) == 10 and len(train) == 40
    assert sorted(train + val) == ids
    assert not set(train) & set(val)
    assert train == sorted(train) and val == sorted(val)


def test_split_depends_on_seed():
    ids = list(range(200))
    assert split_ids(ids, 0.1, 1)[1] != split_ids(ids, 0.1, 2)[1]


def test_split_zero_ratio_and_single_sample():
    assert split_ids([5, 3], 0.0, 1) == ([3, 5], [])
    assert split_ids([9], 0.5, 1) == ([9], [])


def test_jsonl_record():
    rec = format_record("What is 2+2?", None, "4", "open_qa", "jsonl")
    assert rec == {"prompt": "What is 2+2?", "context": None, "response": "4", "category": "open_qa"}
    line = format_line("Résumé ✓", "ctx", "ok\nline2", None, "jsonl")
    assert line.endswith("\n") and line.count("\n") == 1  # embedded newlines are escaped
    assert json.loads(line) == {"prompt": "Résumé ✓", "context": "ctx", "response": "ok\nline2", "category": None}
    assert "Résumé ✓" in line  # not ascii-escaped


def test_chat_record_appends_context_to_user_turn():
    rec = format_record("Summarize this.", "Long text.", "Short.", "summarization", "chat")
    assert rec == {
        "messages": [
            {"role": "user", "content": "Summarize this.\n\nLong text."},
            {"role": "assistant", "content": "Short."},
        ]
    }
    no_ctx = format_record("Hi", "", "Hello", None, "chat")
    assert no_ctx["messages"][0]["content"] == "Hi"


def test_unknown_format_raises():
    with pytest.raises(export_job.ExportError):
        format_record("a", None, "b", None, "csv")


def test_manifest():
    m = build_manifest(
        export_id=12,
        hf_repo_id="databricks/databricks-dolly-15k",
        config="default",
        split="train",
        revision="abc123",
        fields={"prompt": "instruction", "response": "response"},
        filter={"qc_status": ["pass"]},
        fmt="jsonl",
        val_ratio=0.1,
        num_train=9,
        num_val=1,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert m["generator"] == "DataLens"
    assert m["export_id"] == 12
    assert m["source"] == {"hf_repo_id": "databricks/databricks-dolly-15k", "config": "default", "split": "train",
                           "revision": "abc123"}
    assert m["fields"]["prompt"] == "instruction"
    assert m["filter"] == {"qc_status": ["pass"]}
    assert m["counts"] == {"total": 10, "train": 9, "val": 1}
    assert m["format"] == "jsonl" and m["val_ratio"] == 0.1
    assert m["created_at"].startswith("2026-01-02")
    json.dumps(m)


def test_zip_layout(tmp_path):
    train = tmp_path / "t.jsonl"
    val = tmp_path / "v.jsonl"
    train.write_text(format_line("a", None, "b", None, "jsonl") * 3)
    val.write_text("")
    out = tmp_path / "x.zip"
    write_export_zip(str(out), {"generator": "DataLens"}, str(train), str(val))
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["manifest.json", "train.jsonl", "val.jsonl"]
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
        assert json.loads(zf.read("manifest.json")) == {"generator": "DataLens"}
        assert len(zf.read("train.jsonl").splitlines()) == 3
        assert zf.read("val.jsonl") == b""


# ---------- job end-to-end on SQLite ----------
@pytest.fixture
def db_env(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(export_job, "SessionLocal", Session)
    stored: dict[str, str] = {}

    def put_file(key, path, content_type="application/octet-stream"):
        dest = tmp_path / key.replace("/", "_")
        shutil.copy(path, dest)
        stored[key] = str(dest)

    monkeypatch.setattr(export_job.storage, "put_file", put_file)
    return Session, stored


def _seed(Session, n=30, categories=("open_qa", "brainstorming")):
    with Session() as db:
        ds = Dataset(hf_repo_id="databricks/databricks-dolly-15k", config="default", split="train",
                     name="databricks-dolly-15k", revision="rev1", fields={"prompt": "instruction"}, status="ready")
        db.add(ds)
        db.flush()
        db.add_all(
            Sample(
                dataset_id=ds.id, sample_index=i, prompt=f"prompt {i}", context=("ctx" if i % 3 == 0 else None),
                response=f"response {i}", category=categories[i % len(categories)], tokens_est=10 + i,
                qc_status="pass" if i % 2 == 0 else "fail",
            )
            for i in range(n)
        )
        db.commit()
        return ds.id


def _export(Session, dataset_id, filter=None, val_ratio=0.1, fmt="jsonl"):
    with Session() as db:
        e = Export(dataset_id=dataset_id, filter=filter or {}, val_ratio=val_ratio, format=fmt, status="pending")
        db.add(e)
        db.commit()
        return e.id


def _read_zip(path):
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        train = [json.loads(line) for line in zf.read("train.jsonl").decode().splitlines()]
        val = [json.loads(line) for line in zf.read("val.jsonl").decode().splitlines()]
    return manifest, train, val


def test_build_export_jsonl_with_filter(db_env):
    Session, stored = db_env
    ds_id = _seed(Session, n=30)
    eid = _export(Session, ds_id, filter={"qc_status": ["pass"]}, val_ratio=0.2)
    asyncio.run(export_job.build_export({}, eid))
    with Session() as db:
        e = db.get(Export, eid)
        assert e.status == "ready", e.error
        assert e.num_samples == 15 and e.s3_key == f"exports/{eid}.zip"
    manifest, train, val = _read_zip(stored[f"exports/{eid}.zip"])
    assert manifest["counts"] == {"total": 15, "train": 12, "val": 3}
    assert manifest["source"]["hf_repo_id"] == "databricks/databricks-dolly-15k"
    assert manifest["filter"] == {"dataset_id": ds_id, "qc_status": ["pass"]}
    assert len(train) == 12 and len(val) == 3
    for rec in train + val:
        assert set(rec) == {"prompt", "context", "response", "category"}
        assert int(rec["prompt"].split()[1]) % 2 == 0  # only "pass" samples


def test_build_export_is_deterministic(db_env):
    Session, stored = db_env
    ds_id = _seed(Session, n=40)
    eid = _export(Session, ds_id, val_ratio=0.25)
    asyncio.run(export_job.build_export({}, eid))
    _, _, val1 = _read_zip(stored[f"exports/{eid}.zip"])
    asyncio.run(export_job.build_export({}, eid))  # rebuild the same export id
    _, _, val2 = _read_zip(stored[f"exports/{eid}.zip"])
    assert val1 == val2 and len(val1) == 10


def test_build_export_chat_and_failed_check(db_env):
    Session, stored = db_env
    ds_id = _seed(Session, n=12)
    with Session() as db:
        ids = [s.id for s in db.query(Sample).order_by(Sample.sample_index).limit(4)]
        db.add_all(QCResult(sample_id=i, check_name="pii", passed=False, severity="warn", message="x") for i in ids)
        db.add(QCResult(sample_id=ids[0], check_name="formatting", passed=True, severity="pass", message=""))
        db.commit()
    eid = _export(Session, ds_id, filter={"failed_check": "pii"}, val_ratio=0.1, fmt="chat")
    asyncio.run(export_job.build_export({}, eid))
    manifest, train, val = _read_zip(stored[f"exports/{eid}.zip"])
    assert manifest["counts"] == {"total": 4, "train": 3, "val": 1}
    assert manifest["format"] == "chat"
    for rec in train + val:
        assert list(rec) == ["messages"]
        assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    sample0 = next(r for r in train + val if r["messages"][0]["content"].startswith("prompt 0"))
    assert sample0["messages"][0]["content"] == "prompt 0\n\nctx"


def test_build_export_no_matches_fails(db_env):
    Session, _ = db_env
    ds_id = _seed(Session, n=5)
    eid = _export(Session, ds_id, filter={"category": "coding"})
    asyncio.run(export_job.build_export({}, eid))
    with Session() as db:
        e = db.get(Export, eid)
        assert e.status == "failed" and e.error == "no samples match the filter" and e.num_samples == 0


def test_build_export_storage_error_marks_failed(db_env, monkeypatch):
    Session, _ = db_env
    ds_id = _seed(Session, n=5)
    eid = _export(Session, ds_id)

    def boom(*a, **k):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(export_job.storage, "put_file", boom)
    asyncio.run(export_job.build_export({}, eid))
    with Session() as db:
        e = db.get(Export, eid)
        assert e.status == "failed" and "s3 down" in e.error


def test_build_export_missing_export_does_not_crash(db_env):
    asyncio.run(export_job.build_export({}, 12345))


def test_build_export_20k_samples(db_env):
    Session, stored = db_env
    with Session() as db:
        ds = Dataset(hf_repo_id="x/big", name="big", status="ready")
        db.add(ds)
        db.flush()
        db.bulk_insert_mappings(
            Sample,
            [{"dataset_id": ds.id, "sample_index": i, "prompt": f"p{i} " * 20, "response": f"r{i} " * 40,
              "category": "chat", "qc_status": "pass"} for i in range(20000)],
        )
        db.commit()
        ds_id = ds.id
    eid = _export(Session, ds_id, val_ratio=0.1, fmt="chat")
    t0 = time.monotonic()
    asyncio.run(export_job.build_export({}, eid))
    assert time.monotonic() - t0 < 60
    manifest, train, val = _read_zip(stored[f"exports/{eid}.zip"])
    assert manifest["counts"] == {"total": 20000, "train": 18000, "val": 2000}
    assert len(train) == 18000 and len(val) == 2000

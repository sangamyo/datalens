"""Database tables. Schema only — no business logic lives here.

The API contract in docs/api-contract.md is built on these tables; change both together.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Must match the embedding model if semantic search is added (e.g. 384 for bge-small / all-MiniLM-L6-v2).
EMBEDDING_DIM = 384

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("hf_repo_id", "config", "split"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    hf_repo_id: Mapped[str] = mapped_column(String(255))  # e.g. "databricks/databricks-dolly-15k"
    config: Mapped[str] = mapped_column(String(100), default="default")  # HF subset name
    split: Mapped[str] = mapped_column(String(50), default="train")
    name: Mapped[str] = mapped_column(String(255))
    revision: Mapped[str | None] = mapped_column(String(100))  # HF git revision actually imported
    fields: Mapped[dict[str, Any] | None] = mapped_column(JSONType)  # column mapping used, e.g. {"prompt": "instruction", ...}
    num_samples: Mapped[int] = mapped_column(Integer, default=0)  # samples imported (may be < total)
    total_samples: Mapped[int | None] = mapped_column(Integer)  # rows in the source split
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | importing | ready | failed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    samples: Mapped[list["Sample"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (UniqueConstraint("dataset_id", "sample_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    sample_index: Mapped[int] = mapped_column(Integer)  # row number in the source split
    prompt: Mapped[str] = mapped_column(Text, default="")
    context: Mapped[str | None] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str | None] = mapped_column(String(100), index=True)
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0)
    response_chars: Mapped[int] = mapped_column(Integer, default=0)
    tokens_est: Mapped[int] = mapped_column(Integer, default=0)
    lang: Mapped[str | None] = mapped_column(String(8))  # set by QC ("en" or "other")
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)  # normalised prompt+response hash
    qc_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending | pass | warn | fail
    qc_score: Mapped[float | None] = mapped_column(Float)  # 0..1, share of checks passed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset: Mapped[Dataset] = relationship(back_populates="samples")
    qc_results: Mapped[list["QCResult"]] = relationship(back_populates="sample", cascade="all, delete-orphan")


class QCResult(Base):
    __tablename__ = "qc_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id", ondelete="CASCADE"), index=True)
    check_name: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "pii"
    passed: Mapped[bool] = mapped_column(Boolean)
    severity: Mapped[str] = mapped_column(String(20), default="pass")  # pass | warn | fail
    message: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sample: Mapped[Sample] = relationship(back_populates="qc_results")


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    filter: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)  # SampleFilter as JSON
    val_ratio: Mapped[float] = mapped_column(Float, default=0.1)
    format: Mapped[str] = mapped_column(String(20), default="jsonl")  # jsonl | chat
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | building | ready | failed
    num_samples: Mapped[int] = mapped_column(Integer, default=0)
    s3_key: Mapped[str | None] = mapped_column(String(512))  # zip archive
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Embedding(Base):
    """Reserved for semantic search over samples (pgvector)."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id", ondelete="CASCADE"), unique=True)
    model: Mapped[str] = mapped_column(String(100))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

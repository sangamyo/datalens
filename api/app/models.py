"""Database tables. Schema only — no business logic lives here."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Must match the embedding model you pick in Week 6 (e.g. 384 for all-MiniLM-L6-v2).
EMBEDDING_DIM = 384


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    hf_repo_id: Mapped[str] = mapped_column(String(255), unique=True)  # e.g. "lerobot/pusht"
    name: Mapped[str] = mapped_column(String(255))
    fps: Mapped[float | None] = mapped_column(Float)
    robot_type: Mapped[str | None] = mapped_column(String(100))
    num_episodes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="importing")  # importing | ready | failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    episodes: Mapped[list["Episode"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (UniqueConstraint("dataset_id", "episode_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    episode_index: Mapped[int] = mapped_column(Integer)
    length_frames: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float] = mapped_column(Float)
    task: Mapped[str | None] = mapped_column(Text)
    video_key: Mapped[str | None] = mapped_column(String(512))  # object path in S3 storage
    qc_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending | pass | warn | fail
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset: Mapped[Dataset] = relationship(back_populates="episodes")
    qc_results: Mapped[list["QCResult"]] = relationship(back_populates="episode", cascade="all, delete-orphan")


class QCResult(Base):
    __tablename__ = "qc_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    check_name: Mapped[str] = mapped_column(String(64))  # e.g. "timestamp_gap"
    passed: Mapped[bool] = mapped_column(Boolean)
    severity: Mapped[str] = mapped_column(String(20), default="info")  # info | warn | fail
    # JSONB on Postgres, plain JSON elsewhere (keeps the model usable with SQLite if ever needed).
    details: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    episode: Mapped[Episode] = relationship(back_populates="qc_results")


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), unique=True)
    model: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)  # the text that was embedded
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

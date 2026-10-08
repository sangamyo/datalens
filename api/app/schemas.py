"""Pydantic request/response models shared by all routers (see docs/api-contract.md)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QCStatus = Literal["pending", "pass", "warn", "fail"]
CHECK_NAMES = (
    "empty_or_short",
    "length_outlier",
    "exact_duplicate",
    "near_duplicate",
    "pii",
    "non_english",
    "refusal_boilerplate",
    "formatting",
)


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- datasets ----------
class FieldMap(BaseModel):
    prompt: str | None = None
    context: str | None = None
    response: str | None = None
    category: str | None = None
    messages: str | None = None  # chat-format column


class ImportRequest(BaseModel):
    hf_repo_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", examples=["databricks/databricks-dolly-15k"])
    config: str | None = None  # None = the dataset's default config
    split: str = "train"
    max_samples: int = Field(default=2000, ge=1, le=20000)
    fields: FieldMap | None = None  # None = auto-detect


class QCCounts(BaseModel):
    pending: int = 0
    pass_: int = Field(default=0, alias="pass")
    warn: int = 0
    fail: int = 0

    model_config = ConfigDict(populate_by_name=True)


class DatasetOut(ORM):
    id: int
    hf_repo_id: str
    config: str
    split: str
    name: str
    revision: str | None
    fields: dict | None
    num_samples: int
    total_samples: int | None
    status: str
    error: str | None
    created_at: datetime


class CategoryCount(BaseModel):
    name: str
    count: int


class HistogramBucket(BaseModel):
    start: int
    end: int
    count: int


class DatasetDetail(DatasetOut):
    qc_counts: QCCounts
    categories: list[CategoryCount]
    token_histogram: list[HistogramBucket]


# ---------- samples ----------
class SampleOut(ORM):
    id: int
    dataset_id: int
    sample_index: int
    category: str | None
    prompt_preview: str  # first ~160 chars of the prompt
    tokens_est: int
    lang: str | None
    qc_status: str
    qc_score: float | None


class SampleList(BaseModel):
    items: list[SampleOut]
    total: int


class QCResultOut(ORM):
    check_name: str
    passed: bool
    severity: str
    message: str
    details: dict


class SampleDetail(SampleOut):
    prompt: str
    context: str | None
    response: str
    prompt_chars: int
    response_chars: int
    qc_results: list[QCResultOut]
    prev_id: int | None
    next_id: int | None


# ---------- QC ----------
class Enqueued(BaseModel):
    enqueued: int


class QCSummary(BaseModel):
    counts: QCCounts
    by_check: dict[str, dict[str, int]]  # check_name -> {"warn": n, "fail": n}


# ---------- search ----------
class SampleFilter(BaseModel):
    dataset_id: int | None = None
    qc_status: list[QCStatus] | None = None
    category: str | None = None
    lang: str | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    text_contains: str | None = None  # case-insensitive match in prompt, context or response
    failed_check: str | None = None  # samples with a warn/fail result for this check


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    dataset_id: int | None = None
    limit: int = Field(default=50, ge=1, le=500)


class SearchResponse(BaseModel):
    filter: SampleFilter
    parser: Literal["llm", "rules"]
    items: list[SampleOut]
    total: int


# ---------- semantic search (embeddings) ----------
class EmbeddingStatus(BaseModel):
    model: str
    dim: int
    embedded: int  # samples of the dataset with a vector for `model`
    total: int  # samples in the dataset


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    dataset_id: int
    limit: int = Field(default=25, ge=1, le=100)
    filter: SampleFilter = SampleFilter()  # optional structured filter on top; its dataset_id is ignored


class SimilarSample(SampleOut):
    similarity: float  # cosine similarity (1 - cosine distance); 1 = same direction


class SemanticSearchResponse(BaseModel):
    model: str
    items: list[SimilarSample]


# ---------- exports ----------
class ExportCreate(BaseModel):
    dataset_id: int
    filter: SampleFilter = SampleFilter()
    val_ratio: float = Field(default=0.1, ge=0, le=0.5)
    format: Literal["jsonl", "chat"] = "jsonl"


class ExportOut(ORM):
    id: int
    dataset_id: int
    filter: dict
    val_ratio: float
    format: str
    status: str
    num_samples: int
    error: str | None
    created_at: datetime

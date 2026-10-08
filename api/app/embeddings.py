"""Sentence embeddings for semantic search (BAAI/bge-small-en-v1.5 via fastembed / ONNX Runtime).

- 384 dimensions (models.EMBEDDING_DIM), ~67 MB quantized ONNX model, CPU only, no torch.
- The model is downloaded once on first use (to FASTEMBED_CACHE_PATH, a docker volume) and kept
  in memory per process: the worker embeds samples, the API embeds search queries.
- Output vectors are L2-normalised, so cosine distance, inner product and L2 rank the same way.

Tests replace `embed_texts` with a deterministic fake, so they never download the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge's recommended instruction for short queries that retrieve longer passages. Sample-to-sample
# similarity ("similar samples") is symmetric and uses no prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# The model truncates at 512 tokens anyway; cutting the text first saves tokenizer work on huge rows.
MAX_CHARS = 2000
# Texts per ONNX forward pass. Kept small on purpose: attention activations grow with
# batch x heads x seq_len^2, so 256 texts of 512 tokens need ~3 GB (256*12*512^2*4 B), which got the
# worker OOM-killed in a 4 GB VM; 32 stays at a few hundred MB. A batch is padded to its longest
# text, so callers should pass texts sorted by length (see jobs/embed_job.py).
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_model() -> Any:
    from fastembed import TextEmbedding  # heavy import (onnxruntime): only where embeddings are used

    return TextEmbedding(MODEL_NAME)


def sample_text(prompt: str | None, response: str | None) -> str:
    """What we embed for a sample: prompt + response (context is left out: long reference passages
    would fill the 512-token window and drown out the task itself)."""
    text = f"{(prompt or '').strip()}\n\n{(response or '').strip()}".strip()
    return text[:MAX_CHARS]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed passages (sample texts). Returns one 384-dim vector per text, in order."""
    return [v.tolist() for v in get_model().embed(list(texts), batch_size=BATCH_SIZE)]


def embed_query(query: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + query.strip()])[0]

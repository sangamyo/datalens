"""Deterministic stand-in for the embedding model, so tests never download it.

Bag-of-words hashing: each lower-cased word adds +1 to one of 384 dimensions (picked by a stable
hash), then the vector is L2-normalised. Texts that share words point in similar directions, so
nearest-neighbour ranking is predictable.
"""

import hashlib
import math
import re

from app.models import EMBEDDING_DIM


def fake_vector(text: str) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    """Replaces app.embeddings.embed_texts; records the batches it was called with."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts) -> list[list[float]]:
        texts = list(texts)
        self.batches.append(texts)
        return [fake_vector(t) for t in texts]

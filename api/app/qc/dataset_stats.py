"""Dataset-wide inputs for the QC checks, computed once per dataset in (roughly) linear time.

`compute_stats(rows)` takes every sample of a dataset as `SampleRow`s and returns `DatasetStats`:
- token length centre/scale (median/MAD of log1p(tokens_est), with fallbacks) for `length_outlier`
- content-hash groups (first occurrence + later copies) for `exact_duplicate`
- normalised prompt(+context) groups for the "same prompt, different response" warning
- MinHash + LSH near-duplicate pairs (verified with exact Jaccard) for `near_duplicate`

Nothing here touches the database; app.jobs.qc_job loads the rows and stores the results.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------------------------

SHINGLE_SIZE = 5  # word 5-grams of prompt + response
NUM_PERM = 120  # MinHash permutations
LSH_BANDS = 20  # 20 bands x 6 rows: P(candidate | J=0.85) ~ 0.9999, threshold ~ (1/20)^(1/6) = 0.61
LSH_ROWS = NUM_PERM // LSH_BANDS
NEAR_DUP_MIN_JACCARD = 0.85  # pairs below this exact Jaccard are not recorded
NEAR_DUP_MIN_RESPONSE_SIM = 0.5  # ... and the responses must share >= 50% of their words
NEAR_DUP_TOP_K = 5  # similar_to keeps the best K matches
MAX_BUCKET_PAIRS = 50  # a huge LSH bucket only compares each member with its first 50 bucket-mates
MINHASH_CHUNK = 4_096  # shingles hashed per numpy chunk (chunk x NUM_PERM uint64 ~ 4 MB, cache friendly)
MAX_GROUP_IDS = 10  # duplicate_of / same_prompt_as lists are capped at this many ids
SEED = 1234567

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class SampleRow:
    id: int
    sample_index: int
    prompt: str
    context: str | None
    response: str
    tokens_est: int
    content_hash: str | None = None


@dataclass
class DatasetStats:
    n: int
    token_center: float | None  # median of log1p(tokens_est)
    token_scale: float | None  # robust std of log1p(tokens_est); None/0 -> length check skipped
    token_scale_method: str
    token_median: float | None  # median tokens_est (for messages)
    hash_of: dict[int, str] = field(default_factory=dict)  # sample id -> effective content hash
    hash_groups: dict[str, list[int]] = field(default_factory=dict)  # hash -> ids in sample_index order
    duplicate_of: dict[int, list[int]] = field(default_factory=dict)  # later copy id -> earlier copy ids
    prompt_conflicts: dict[int, list[int]] = field(default_factory=dict)  # id -> same prompt, other response
    near_dups: dict[int, list[tuple[int, float]]] = field(default_factory=dict)  # id -> [(other id, jaccard)]
    index_of: dict[int, int] = field(default_factory=dict)  # sample id -> sample_index


# ---------------------------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------------------------


def normalize_text(text: str | None) -> str:
    """Lower-case, drop punctuation, collapse whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", _NON_WORD_RE.sub(" ", text.lower())).strip()


def fallback_content_hash(prompt: str | None, context: str | None, response: str | None) -> str:
    """Used when the importer did not store Sample.content_hash."""
    key = "\x1f".join(_WS_RE.sub(" ", (t or "").lower()).strip() for t in (prompt, context, response))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Length statistics
# ---------------------------------------------------------------------------------------------


def robust_center_scale(values: Sequence[float] | np.ndarray) -> tuple[float | None, float | None, str]:
    """Median and a robust std estimate: MAD, falling back to IQR, then mean absolute deviation."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return None, None, "none"
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        return med, 1.4826 * mad, "mad"
    q1, q3 = np.percentile(x, [25, 75])
    if q3 - q1 > 0:
        return med, float(q3 - q1) / 1.349, "iqr"
    mean_abs = float(np.mean(np.abs(x - med)))
    if mean_abs > 0:
        return med, 1.2533 * mean_abs, "meanabs"
    return med, 0.0, "constant"


# ---------------------------------------------------------------------------------------------
# MinHash / LSH
# ---------------------------------------------------------------------------------------------


def _splitmix(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser (uint64 in, uint64 out; wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


_rng = np.random.default_rng(SEED)
_SHINGLE_MULT = (_rng.integers(1, 2**63, size=SHINGLE_SIZE, dtype=np.uint64) * np.uint64(2) + np.uint64(1)).astype(
    np.uint64
)
_PERM_A = (_rng.integers(1, 2**63, size=NUM_PERM, dtype=np.uint64) * np.uint64(2) + np.uint64(1)).astype(np.uint64)
_PERM_B = _rng.integers(0, 2**63, size=NUM_PERM, dtype=np.uint64)
_BAND_MULT = (_rng.integers(1, 2**63, size=LSH_ROWS, dtype=np.uint64) * np.uint64(2) + np.uint64(1)).astype(np.uint64)


class _WordHasher:
    def __init__(self) -> None:
        self.cache: dict[str, int] = {}

    def __call__(self, words: list[str]) -> np.ndarray:
        cache = self.cache
        out = np.empty(len(words), dtype=np.uint64)
        for i, w in enumerate(words):
            h = cache.get(w)
            if h is None:
                h = zlib.crc32(w.encode("utf-8")) | (zlib.adler32(w.encode("utf-8")) << 32)
                cache[w] = h
            out[i] = h
        return out


def shingle_hashes(text: str, hasher: _WordHasher | None = None) -> np.ndarray:
    """Sorted unique uint64 hashes of the word 5-shingles of `text` (one shingle if < 5 words)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return np.empty(0, dtype=np.uint64)
    w = (hasher or _WordHasher())(words)
    k = SHINGLE_SIZE
    with np.errstate(over="ignore"):
        if len(w) < k:
            h = np.array([np.sum(w * _SHINGLE_MULT[: len(w)], dtype=np.uint64)], dtype=np.uint64)
        else:
            m = len(w) - k + 1
            h = np.zeros(m, dtype=np.uint64)
            for j in range(k):
                h += w[j : j + m] * _SHINGLE_MULT[j]
    return np.unique(_splitmix(h))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def minhash_signatures(shingles: list[np.ndarray]) -> np.ndarray:
    """(n, NUM_PERM) uint64 MinHash signatures; rows for empty shingle sets are all-max."""
    n = len(shingles)
    sig = np.full((n, NUM_PERM), np.iinfo(np.uint64).max, dtype=np.uint64)
    nonempty = [i for i, s in enumerate(shingles) if s.size]
    start = 0
    while start < len(nonempty):
        # Take whole samples until the chunk holds ~MINHASH_CHUNK shingles (at least one sample).
        end, total = start, 0
        while end < len(nonempty) and (end == start or total + shingles[nonempty[end]].size <= MINHASH_CHUNK):
            total += shingles[nonempty[end]].size
            end += 1
        idx = nonempty[start:end]
        flat = np.concatenate([shingles[i] for i in idx])
        sizes = np.array([shingles[i].size for i in idx])
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        with np.errstate(over="ignore"):
            vals = flat[:, None] * _PERM_A[None, :] + _PERM_B[None, :]  # shingles are already mixed
        sig[idx] = np.minimum.reduceat(vals, offsets, axis=0)
        start = end
    return sig


def lsh_candidate_pairs(sig: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Candidate (i, j) pairs (i < j, positions into `sig`) sharing at least one LSH band."""
    pos = np.flatnonzero(active)
    if pos.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs: list[np.ndarray] = []
    for b in range(LSH_BANDS):
        band = sig[pos, b * LSH_ROWS : (b + 1) * LSH_ROWS]
        with np.errstate(over="ignore"):
            keys = _splitmix((band * _BAND_MULT[None, :]).sum(axis=1, dtype=np.uint64) + np.uint64(b))
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        # boundaries of runs of equal keys
        change = np.flatnonzero(np.diff(sk) != 0) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [sk.size]])
        multi = np.flatnonzero(ends - starts > 1)
        for r in multi:
            members = np.sort(pos[order[starts[r] : ends[r]]])
            m = members.size
            if m <= MAX_BUCKET_PAIRS:
                ii, jj = np.triu_indices(m, k=1)
            else:  # pathological bucket: compare everyone with the first MAX_BUCKET_PAIRS members only
                ii, jj = np.meshgrid(np.arange(MAX_BUCKET_PAIRS), np.arange(m), indexing="ij")
                keep = jj > ii
                ii, jj = ii[keep], jj[keep]
            pairs.append(np.stack([members[ii], members[jj]], axis=1))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    allp = np.concatenate(pairs).astype(np.int64)
    return np.unique(allp, axis=0)


def _word_set(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def near_duplicate_map(
    texts: Sequence[str],
    ids: Sequence[int],
    threshold: float = NEAR_DUP_MIN_JACCARD,
    responses: Sequence[str] | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """id -> [(other id, jaccard)] for every pair with exact shingle Jaccard >= threshold (both directions).

    With `responses`, a pair also needs response word-set Jaccard >= NEAR_DUP_MIN_RESPONSE_SIM: two
    different answers to the same long prompt are a "same prompt" case, not a near-duplicate.
    """
    hasher = _WordHasher()
    shingles = [shingle_hashes(t, hasher) for t in texts]
    sig = minhash_signatures(shingles)
    active = np.array([s.size > 0 for s in shingles], dtype=bool)
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, j in lsh_candidate_pairs(sig, active):
        jac = jaccard(shingles[i], shingles[j])
        if jac >= threshold and responses is not None:
            ri, rj = _word_set(responses[i]), _word_set(responses[j])
            union = len(ri | rj)
            if union and len(ri & rj) / union < NEAR_DUP_MIN_RESPONSE_SIM:
                continue
        if jac >= threshold:
            out[ids[i]].append((ids[j], round(jac, 4)))
            out[ids[j]].append((ids[i], round(jac, 4)))
    for k in out:
        out[k].sort(key=lambda p: (-p[1], p[0]))
    return dict(out)


# ---------------------------------------------------------------------------------------------
# One-pass dataset statistics
# ---------------------------------------------------------------------------------------------


def compute_stats(rows: Iterable[SampleRow]) -> DatasetStats:
    rows = sorted(rows, key=lambda r: (r.sample_index, r.id))
    n = len(rows)
    tokens = np.fromiter((max(int(r.tokens_est or 0), 0) for r in rows), dtype=np.float64, count=n)
    center, scale, method = robust_center_scale(np.log1p(tokens))
    stats = DatasetStats(
        n=n,
        token_center=center,
        token_scale=scale,
        token_scale_method=method,
        token_median=float(np.median(tokens)) if n else None,
    )

    hash_groups: dict[str, list[int]] = defaultdict(list)
    prompt_groups: dict[str, list[int]] = defaultdict(list)
    rep_ids: list[int] = []
    rep_texts: list[str] = []
    rep_responses: list[str] = []
    for r in rows:
        h = r.content_hash or fallback_content_hash(r.prompt, r.context, r.response)
        stats.hash_of[r.id] = h
        stats.index_of[r.id] = r.sample_index
        if h not in hash_groups:  # only the first copy of an exact duplicate takes part in near-dup search
            rep_ids.append(r.id)
            rep_texts.append(f"{r.prompt or ''}\n{r.response or ''}")
            rep_responses.append(r.response or "")
        hash_groups[h].append(r.id)
        pkey = normalize_text(r.prompt) + "\x1f" + normalize_text(r.context)
        if pkey.strip("\x1f"):
            prompt_groups[pkey].append(r.id)
    stats.hash_groups = dict(hash_groups)
    for members in hash_groups.values():
        for k in range(1, len(members)):
            stats.duplicate_of[members[k]] = members[: min(k, MAX_GROUP_IDS)]

    for members in prompt_groups.values():
        if len(members) < 2:
            continue
        for sid in members:
            others = [o for o in members if stats.hash_of[o] != stats.hash_of[sid]]
            if others:
                stats.prompt_conflicts[sid] = others[:MAX_GROUP_IDS]

    stats.near_dups = near_duplicate_map(rep_texts, rep_ids, responses=rep_responses)
    return stats

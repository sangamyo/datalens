"""Dataset-wide QC stats: MinHash/LSH near duplicates and performance."""

import random
import re
import time

from app.qc.checks import run_all
from app.qc.dataset_stats import SampleRow, compute_stats, jaccard, near_duplicate_map, shingle_hashes

BASE = (
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize "
    "foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates "
    "oxygen as a byproduct, which is released into the atmosphere through small pores in the leaves. "
    "The light-dependent reactions take place in the thylakoid membranes, where water is split and energy is "
    "captured as ATP and NADPH, while the Calvin cycle in the stroma uses that energy to fix carbon into sugars "
    "that the plant later uses for growth, storage and respiration."
)


def test_shingles_and_jaccard() -> None:
    a = shingle_hashes(BASE)
    assert a.size == len(re.findall(r"\w+", BASE)) - 4  # all 5-shingles are distinct here
    assert jaccard(a, shingle_hashes(BASE.upper())) == 1.0  # case-insensitive
    assert jaccard(a, shingle_hashes("Completely unrelated text about football and the world cup final.")) == 0.0
    assert shingle_hashes("").size == 0
    assert shingle_hashes("two words").size == 1


def test_near_duplicate_map_finds_near_copies_only() -> None:
    near = BASE.replace("small pores", "tiny pores")  # one word changed
    unrelated = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after "
        "the engineer Gustave Eiffel, whose company designed and built the tower for the 1889 World's Fair."
    )
    rewrite = BASE.split(". ")[0] + ". Plants are green and need water to live."  # shares only the first sentence
    m = near_duplicate_map([BASE, near, unrelated, rewrite], [10, 11, 12, 13])
    assert set(m) == {10, 11}
    (other, jac), = m[10]
    assert other == 11 and 0.85 <= jac < 1.0


def test_different_answers_to_same_long_prompt_are_not_near_duplicates() -> None:
    story = BASE + " Summarise it."
    m = near_duplicate_map(
        [story + "\nPlants turn light into sugar.", story + "\nGreen leaves make food and oxygen from sunlight."],
        [1, 2],
        responses=["Plants turn light into sugar.", "Green leaves make food and oxygen from sunlight."],
    )
    assert m == {}


def test_near_duplicate_check_on_dataset() -> None:
    rows = [
        SampleRow(1, 0, "Explain photosynthesis.", None, BASE, 80),
        SampleRow(2, 1, "What is the Eiffel Tower?", None, "A famous iron tower in Paris built for the 1889 fair.", 20),
        SampleRow(3, 2, "Explain photosynthesis.", None, BASE.replace("small pores", "tiny pores"), 80),
    ]
    stats = compute_stats(rows)
    res = {r.id: {c.check_name: c for c in run_all(r, stats)} for r in rows}
    assert res[1]["near_duplicate"].severity == "pass"
    assert res[2]["near_duplicate"].severity == "pass"
    nd = res[3]["near_duplicate"]
    assert nd.severity == "warn" and nd.details["similar_to"][0]["id"] == 1


def _synthetic(n: int, seed: int = 0) -> list[SampleRow]:
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(8000)] + ["the", "a", "of", "and", "to", "is", "in"] * 300
    rows = []
    for i in range(n):
        p = " ".join(rnd.choices(vocab, k=rnd.randint(5, 25)))
        r = " ".join(rnd.choices(vocab, k=rnd.randint(10, 150)))
        rows.append(SampleRow(i + 1, i, p, None, r, (len(p) + len(r)) // 4))
    return rows


def test_near_duplicate_and_stats_scale_to_20k() -> None:
    rows = _synthetic(20_000)
    # plant 40 near copies (one word appended) and 40 exact copies
    planted = []
    for k in range(40):
        src = rows[k * 100]
        rid = 100_000 + k
        rows.append(SampleRow(rid, 50_000 + k, src.prompt, None, src.response + " extra", src.tokens_est))
        planted.append((rid, src.id))
        rows.append(SampleRow(200_000 + k, 60_000 + k, src.prompt, None, src.response, src.tokens_est))
    t0 = time.perf_counter()
    stats = compute_stats(rows)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10, f"compute_stats took {elapsed:.1f}s"
    found = sum(1 for rid, src in planted if any(o == src for o, _ in stats.near_dups.get(rid, [])))
    assert found >= 38  # LSH recall at J>=0.85 is ~1, allow a tiny miss rate
    # random texts must not be paired with each other
    false_pairs = [i for i, v in stats.near_dups.items() if i < 100_000 and all(o < 100_000 for o, _ in v)]
    assert not false_pairs
    assert len(stats.duplicate_of) == 40
    assert stats.token_scale and stats.token_scale > 0

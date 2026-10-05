"""Unit tests for the pure QC checks (no Docker, no DB)."""

import pytest

from app.qc import checks
from app.qc.checks import (
    CheckResult,
    check_empty_or_short,
    check_exact_duplicate,
    check_formatting,
    check_length_outlier,
    check_near_duplicate,
    check_non_english,
    check_pii,
    check_refusal_boilerplate,
    find_pii,
    run_all,
    summarize,
)
from app.qc.dataset_stats import SampleRow, compute_stats
from app.schemas import CHECK_NAMES

CLEAN_PROMPT = "What are the main causes of the French Revolution?"
CLEAN_RESPONSE = (
    "The French Revolution was caused by a mix of fiscal crisis, social inequality between the three estates, "
    "Enlightenment ideas that questioned absolute monarchy, and poor harvests that drove up bread prices in 1788 and 1789."
)

# Realistic Dolly-style clean samples that must not trigger anything.
CLEAN_SAMPLES = [
    (CLEAN_PROMPT, None, CLEAN_RESPONSE),
    (
        "When did Virgin Australia start operating?",
        "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. "
        "It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
        "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
    ),
    (
        "How many people live in Tokyo?",
        None,
        "The Greater Tokyo Area has a population of about 37,468,000 people, making it the most populous "
        "metropolitan area in the world. The city proper had 13,960,000 residents in 2021.",
    ),
    (
        "Write a Python function that returns the square of a number.",
        None,
        "Here is a simple function:\n\n```python\ndef square(x):\n    return x * x\n```\n\nCall it as `square(4)`, which returns 16.",
    ),
    (
        "List three primary colors.",
        None,
        "- Red\n- Yellow\n- Blue",
    ),
]


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.check_name: r for r in results}


# ---------------------------------------------------------------- empty_or_short
def test_empty_or_short_levels() -> None:
    assert check_empty_or_short(CLEAN_PROMPT, CLEAN_RESPONSE).severity == "pass"
    assert check_empty_or_short(CLEAN_PROMPT, "   \n").severity == "fail"
    assert check_empty_or_short("", CLEAN_RESPONSE).severity == "fail"
    short = check_empty_or_short(CLEAN_PROMPT, "Paris.")
    assert short.severity == "warn" and "1 words" in short.message
    assert check_empty_or_short(CLEAN_PROMPT, "It is a cat").severity == "warn"  # 11 chars < 15
    assert check_empty_or_short(CLEAN_PROMPT, "Paris.", min_words=1, min_chars=1).severity == "pass"


# ---------------------------------------------------------------- length_outlier
def test_length_outlier() -> None:
    import math

    center, scale = math.log1p(100), 0.5
    assert check_length_outlier(110, center, scale, 500, 100).severity == "pass"
    assert check_length_outlier(100 * 6, center, scale, 500, 100).severity == "warn"  # z ~ 3.6
    assert check_length_outlier(100 * 20, center, scale, 500, 100).severity == "fail"  # z ~ 6.0
    assert check_length_outlier(1, center, scale, 500, 100).severity == "fail"  # extremely short
    skipped = check_length_outlier(5000, center, scale, 10, 100)
    assert skipped.severity == "pass" and skipped.message.startswith("skipped: fewer than 20 samples")


def test_robust_scale_falls_back_when_mad_is_zero() -> None:
    from app.qc.dataset_stats import robust_center_scale

    vals = [5.0] * 60 + [6.0] * 30 + [50.0] * 10  # MAD 0, IQR > 0
    c, s, method = robust_center_scale(vals)
    assert c == 5.0 and s > 0 and method == "iqr"
    c, s, method = robust_center_scale([5.0] * 95 + [9.0] * 5)  # MAD 0, IQR 0
    assert s > 0 and method == "meanabs"
    assert robust_center_scale([3.0] * 10)[2] == "constant"


# ---------------------------------------------------------------- duplicates
def test_exact_duplicate_levels() -> None:
    assert check_exact_duplicate([]).severity == "pass"
    r = check_exact_duplicate([11, 12])
    assert r.severity == "fail" and r.details["duplicate_of"] == [11, 12]
    w = check_exact_duplicate([], [7])
    assert w.severity == "warn" and w.details["same_prompt_as"] == [7]


def test_near_duplicate_levels() -> None:
    assert check_near_duplicate([]).severity == "pass"
    w = check_near_duplicate([(3, 0.9), (4, 0.86)])
    assert w.severity == "warn" and w.details["similar_to"][0] == {"id": 3, "jaccard": 0.9}
    assert check_near_duplicate([(3, 0.97)]).severity == "fail"
    assert check_near_duplicate([(3, 0.97)], is_exact_duplicate=True).severity == "warn"
    first = check_near_duplicate([], [(9, 0.9)])
    assert first.severity == "pass" and first.details["similar_to"] == [{"id": 9, "jaccard": 0.9}]


def _row(i: int, prompt: str, response: str, context: str | None = None) -> SampleRow:
    return SampleRow(i, i, prompt, context, response, (len(prompt) + len(response)) // 4)


def test_run_all_duplicates_first_occurrence_passes() -> None:
    rows = [
        _row(1, CLEAN_PROMPT, CLEAN_RESPONSE),
        _row(2, "Name a fruit that is yellow and long.", "A banana is a long yellow fruit that grows in bunches."),
        _row(3, CLEAN_PROMPT, CLEAN_RESPONSE),  # exact copy of 1
        _row(4, CLEAN_PROMPT, "It was mostly about taxes, bread prices and new political ideas."),  # same prompt
    ]
    stats = compute_stats(rows)
    r1, r3, r4 = (by_name(run_all(r, stats)) for r in (rows[0], rows[2], rows[3]))
    assert r1["exact_duplicate"].severity == "warn"  # same prompt as #4 with a different response
    assert r1["exact_duplicate"].details["same_prompt_as"] == [4]
    assert r3["exact_duplicate"].severity == "fail" and r3["exact_duplicate"].details["duplicate_of"] == [1]
    assert r4["exact_duplicate"].severity == "warn"
    assert r3["near_duplicate"].severity == "pass"  # exact copies are reported by exact_duplicate only


# ---------------------------------------------------------------- pii
@pytest.mark.parametrize(
    "text,kind",
    [
        ("Contact me at priya.sharma@example.co.in for details.", "email"),
        ("Call me on +91 98765 43210 tomorrow.", "phone"),
        ("My number is 9876543210.", "phone"),
        ("Reach our US office at (415) 555-2671.", "phone"),
        ("Dial +44 20 7946 0958 for London.", "phone"),
        ("Server is at 203.0.113.42 behind the proxy.", "ip_address"),
        ("Card: 4111 1111 1111 1111 exp 12/27", "credit_card"),
        ("Aadhaar 2345 6789 0123 belongs to him.", "aadhaar"),
        ("export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456", "secret"),
        ("aws key AKIAIOSFODNN7EXAMPLE leaked", "secret"),
    ],
)
def test_pii_detects(text: str, kind: str) -> None:
    spans = find_pii(text, "response")
    assert [s["kind"] for s in spans] == [kind]
    s = spans[0]
    assert s["field"] == "response" and 0 <= s["start"] < s["end"] <= len(text)


def test_pii_offsets_and_fields() -> None:
    r = check_pii("Email bob@corp.com please", "ctx 10.1.2.3", "ok")
    assert r.severity == "fail" or r.severity == "warn"
    spans = r.details["spans"]
    assert {s["field"] for s in spans} == {"prompt", "context"}
    p = next(s for s in spans if s["field"] == "prompt")
    assert "Email bob@corp.com please"[p["start"] : p["end"]] == "bob@corp.com"


def test_pii_severity() -> None:
    assert check_pii("Who wrote it?", None, "Mail jane@example.org").severity == "warn"
    assert check_pii("Pay", None, "Use card 4111-1111-1111-1111").severity == "fail"
    assert check_pii("x", None, "a@b.com, c@d.com and e@f.com").severity == "fail"  # >= 3 hits


@pytest.mark.parametrize(
    "text",
    [
        "The war lasted from 1939 to 1945 and killed over 70,000,000 people.",
        "Tokyo has 13960000 residents and an area of 2194 km2.",
        "Python 3.11.4 was released in 2023; numpy 1.26.0 too.",
        "The ISBN is 978-3-16-148410-0 and the price is $1,299.99.",
        "Order 1234567890123 was shipped.",  # 13 digits but fails Luhn
        "Set the timeout to 30000 ms and retry 5 times.",
        "The coordinates are 40.7128, -74.0060.",
        "Version 1.2.3.4 of the firmware",
        "Call the function f(x) = 3x + 2 and print(f(10)).",
        "He was born on 12-05-1990 in Mumbai.",
    ],
)
def test_pii_no_false_positives(text: str) -> None:
    assert find_pii(text, "response") == [], find_pii(text, "response")


def test_luhn() -> None:
    assert checks._luhn_ok("4111111111111111")
    assert not checks._luhn_ok("4111111111111112")


# ---------------------------------------------------------------- non_english
def test_non_english() -> None:
    en = check_non_english(CLEAN_PROMPT, None, CLEAN_RESPONSE)
    assert en.severity == "pass" and en.details["lang"] == "en"
    hi = check_non_english("भारत की राजधानी क्या है?", None, "भारत की राजधानी नई दिल्ली है। यह एक बड़ा शहर है।")
    assert hi.severity == "warn" and hi.details["lang"] == "other"
    es = check_non_english(
        "¿Cuál es la capital de España?",
        None,
        "La capital de España es Madrid, que es también la ciudad más grande del país y una de las más visitadas de Europa.",
    )
    assert es.severity == "warn" and es.details["lang"] == "other"
    # Spanish place names inside an English answer are not a foreign-language sample.
    places = check_non_english(
        "Tell me if the following attractions in Barcelona are free or paid: Park Guell, Parc de la Ciutadella",
        None,
        "Park Guell - paid, Parc de la Ciutadella - free, La Boqueria - free, La Rambla - free, Camp Nou - paid",
    )
    assert places.severity == "pass"
    zh = check_non_english("中国的首都是哪里？", None, "中国的首都是北京，它是一个历史悠久的城市。")
    assert zh.details["lang"] == "other"


def test_non_english_ignores_code_and_names() -> None:
    code = check_non_english(
        "Fix this",
        None,
        "```js\nconst x = {a: [1,2,3]}; if (x.a.length > 2) { console.log(x); }\n```",
    )
    assert code.severity == "pass" and code.details["lang"] == "en"
    names = check_non_english(
        "Who was Gabriel García Márquez?",
        None,
        "Gabriel García Márquez was a Colombian novelist who wrote One Hundred Years of Solitude and won the Nobel Prize.",
    )
    assert names.severity == "pass"
    translate = check_non_english(
        "Translate 'good morning, how are you today?' into French.", None, "Bonjour, comment allez-vous aujourd'hui ?"
    )
    assert translate.severity == "pass"


# ---------------------------------------------------------------- refusal_boilerplate
def test_refusal() -> None:
    assert check_refusal_boilerplate(CLEAN_PROMPT, CLEAN_RESPONSE).severity == "pass"
    r = check_refusal_boilerplate("How do I pick a lock?", "I'm sorry, but I can't help with that request.")
    assert r.severity == "fail"
    long_resp = "As an AI language model, I don't have personal opinions. " + "However, here is some context. " * 20
    w = check_refusal_boilerplate("What's the best pizza?", long_resp)
    assert w.severity == "warn" and w.details["phrases"]
    v = check_refusal_boilerplate("Write a poem", "I am ChatGPT and here is a poem for you. " + "Roses are red. " * 10)
    assert v.severity == "warn" and v.details["vendor_mentions"] == 1
    # Encyclopedic mention in an answer about another product is not boilerplate.
    copilot = "Copilot uses OpenAI's GPT-4 large language model with Microsoft Graph to assist users in many tasks."
    assert check_refusal_boilerplate("What is Microsoft Copilot?", copilot).severity == "pass"
    # The prompt asks about OpenAI: mentioning it is fine.
    assert check_refusal_boilerplate("What is OpenAI?", "OpenAI is an AI research company founded in 2015.").severity == "pass"
    # Ordinary apologies / "cannot" in a normal answer are fine.
    assert (
        check_refusal_boilerplate("Can penguins fly?", "No, penguins cannot fly; their wings evolved into flippers.").severity
        == "pass"
    )


# ---------------------------------------------------------------- formatting
def test_formatting_clean() -> None:
    for prompt, _ctx, resp in CLEAN_SAMPLES:
        assert check_formatting(prompt, resp).severity == "pass", resp


def test_formatting_issues() -> None:
    fence = check_formatting("Write code", "Here:\n```python\nprint('hi')\n")
    assert fence.severity == "warn" and fence.details["issues"] == ["unbalanced_code_fence"]
    brackets = check_formatting("Write code", "```python\ndef f(x:\n    return [x, (x + 1]\n```")
    assert brackets.details["issues"] == ["unbalanced_brackets"]
    rep = check_formatting("Say hi", "Hiiiiiiiiiiiiiiiiiiiiiiiiiiiii there friend")
    assert rep.details["issues"] == ["repeated_characters"]
    prompt = "Explain why the sky appears blue during the day and red at sunset"
    echo = check_formatting(prompt, prompt + ".")
    assert "prompt_echo" in echo.details["issues"]
    words = " ".join(["the quick brown fox jumps over the lazy dog"] * 7)
    trunc = check_formatting("Tell a story", words.capitalize() + " and then the fox decided to walk to the")
    assert trunc.details["issues"] == ["truncated"]
    cite = check_formatting("Tell a story", words.capitalize() + ". It was a long day.[3")
    assert cite.details["issues"] == ["truncated"]
    # Human answers often just omit the final full stop: not truncation.
    assert check_formatting("Tell a story", words.capitalize() + " and the fox went home happy").severity == "pass"
    # Restating the question in the answer is not an echo.
    q = "The BCG vaccine is administered to Indian children soon after birth to protect them against which disease?"
    a = "The BCG vaccine is administered to Indian children soon after birth to protect them against tuberculosis"
    assert check_formatting(q, a).severity == "pass"
    multi = check_formatting("Write code", "``` \nxxxxxxxxxxxxxxxxxxxxxxxxx\n")
    assert multi.severity == "fail" and len(multi.details["issues"]) >= 2


def test_formatting_brackets_in_strings_ok() -> None:
    resp = "```python\nprint(\"(not closed\")\nx = [1, 2]  # ) comment\n```"
    assert check_formatting("code", resp).severity == "pass"


# ---------------------------------------------------------------- whole sample + summarize
def test_clean_samples_pass_everything() -> None:
    rows = [_row(i, p, r, c) for i, (p, c, r) in enumerate(CLEAN_SAMPLES)]
    stats = compute_stats(rows)
    for row in rows:
        results = run_all(row, stats)
        assert [r.check_name for r in results] == list(CHECK_NAMES)
        bad = [(r.check_name, r.message) for r in results if r.severity != "pass" and r.check_name != "empty_or_short"]
        assert not bad, (row.prompt, bad)
        assert all(isinstance(r.message, str) and "\n" not in r.message for r in results)


def test_results_are_json_safe() -> None:
    import json

    rows = [_row(1, "Email me", "Sure: a@b.com " * 3), _row(2, "Email me", "Sure: a@b.com " * 3)]
    stats = compute_stats(rows)
    for row in rows:
        for r in run_all(row, stats):
            json.dumps(r.details)


def test_summarize() -> None:
    p = CheckResult("a", True, "pass", "", {})
    w = CheckResult("b", False, "warn", "", {})
    f = CheckResult("c", False, "fail", "", {})
    assert summarize([p, p]) == ("pass", 1.0)
    assert summarize([p, w, p, p]) == ("warn", 0.75)
    assert summarize([p, w, f, p]) == ("fail", 0.5)
    assert summarize([]) == ("pending", None)

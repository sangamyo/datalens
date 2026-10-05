"""Quality checks for instruction / chat fine-tuning samples.

Every check is a pure function: plain strings / numbers in, a `CheckResult` out. No I/O and no
database — dataset-wide inputs come from app.qc.dataset_stats, persistence from app.jobs.qc_job.

Severity convention: "pass" (fine), "warn" (worth a look), "fail" (likely harmful for training).
`passed` is True only for severity "pass". A check that cannot run (e.g. too few samples) returns
severity "pass" with a message starting with "skipped:".
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.schemas import CHECK_NAMES

if TYPE_CHECKING:
    from app.qc.dataset_stats import DatasetStats, SampleRow

__all__ = ["CHECK_NAMES", "CheckResult", "run_all", "summarize"]

# ---------------------------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------------------------

# empty_or_short
SHORT_RESPONSE_MIN_WORDS = 3  # response with fewer words -> warn
SHORT_RESPONSE_MIN_CHARS = 15  # ... or fewer non-space-trimmed characters -> warn

# length_outlier: robust z-score of log1p(tokens_est) vs the dataset (lengths are log-normal-ish;
# on the raw scale every long-form answer in a mostly-short dataset would be an "outlier").
LENGTH_MIN_SAMPLES = 20  # smaller datasets are skipped
# Tuned on dolly-15k (median 114 tokens): z > 3.0 ~ 0.1% of samples (> ~3,000 tokens), z > 4.5 is extreme.
LENGTH_WARN_Z = 3.0
LENGTH_FAIL_Z = 4.5

# exact_duplicate / near_duplicate
NEAR_DUP_WARN_JACCARD = 0.85  # shingle Jaccard >= this with an earlier sample -> warn
NEAR_DUP_FAIL_JACCARD = 0.95  # >= this -> fail (unless already an exact duplicate)
SIMILAR_TOP_K = 5

# pii
PII_FAIL_HITS = 3  # this many hits of any kind -> fail
PII_HIGH_RISK = {"credit_card", "aadhaar", "secret"}  # any one of these -> fail

# non_english
LANG_MIN_LETTERS = 20  # fewer letters than this -> too short to judge (lang "en", pass)
LANG_MIN_ASCII_SHARE = 0.70  # share of letters that are ASCII Latin; below -> other
LANG_MIN_WORDS_FOR_STOPWORDS = 12  # stopword test needs at least this many words
LANG_MAX_FOREIGN_RATIO = 0.12  # foreign-stopword share above this AND above the English share -> other
LANG_MAX_EN_RATIO_FOR_FOREIGN = 0.06  # real foreign prose has almost no English function words (lists of
# Spanish/French place names inside English answers have some)
CODE_SYMBOL_DENSITY = 0.08  # share of code symbols ({}();=<>[]#$_ etc.) that marks text as code

# refusal_boilerplate
REFUSAL_FAIL_MAX_WORDS = 60  # a response this short that contains a refusal phrase is "mostly refusal"

# formatting
REPEATED_CHAR_RUN = 20  # >= this many identical non-space characters in a row
PROMPT_ECHO_MIN_WORDS = 8  # responses shorter than this are not tested for echo
PROMPT_ECHO_CONTAINMENT = 0.9  # share of response word-trigrams found in the prompt ...
PROMPT_ECHO_COVERAGE = 0.95  # ... and share of prompt word-trigrams found in the response
TRUNCATION_MIN_WORDS = 50  # only long responses are tested for trailing truncation


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    severity: str  # pass | warn | fail
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _result(name: str, severity: str, message: str, details: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, severity == "pass", severity, message, details or {})


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str | None) -> list[str]:
    return _WORD_RE.findall(text or "")


# ---------------------------------------------------------------------------------------------
# empty_or_short
# ---------------------------------------------------------------------------------------------


def check_empty_or_short(
    prompt: str | None,
    response: str | None,
    min_words: int = SHORT_RESPONSE_MIN_WORDS,
    min_chars: int = SHORT_RESPONSE_MIN_CHARS,
) -> CheckResult:
    name = "empty_or_short"
    p, r = (prompt or "").strip(), (response or "").strip()
    if not r and not p:
        return _result(name, "fail", "prompt and response are empty", {"prompt_chars": 0, "response_chars": 0})
    if not r:
        return _result(name, "fail", "response is empty", {"prompt_chars": len(p), "response_chars": 0})
    if not p:
        return _result(name, "fail", "prompt is empty", {"prompt_chars": 0, "response_chars": len(r)})
    nw = len(r.split())
    details = {"response_words": nw, "response_chars": len(r)}
    if nw < min_words or len(r) < min_chars:
        return _result(name, "warn", f"short response: {nw} words, {len(r)} chars (min {min_words} / {min_chars})", details)
    return _result(name, "pass", f"response has {nw} words, {len(r)} chars", details)


# ---------------------------------------------------------------------------------------------
# length_outlier
# ---------------------------------------------------------------------------------------------


def check_length_outlier(
    tokens_est: int,
    center: float | None,
    scale: float | None,
    n: int,
    median_tokens: float | None = None,
    method: str = "mad",
) -> CheckResult:
    """`center`/`scale` are the robust centre/std of log1p(tokens_est) over the dataset."""
    name = "length_outlier"
    if n < LENGTH_MIN_SAMPLES:
        return _result(name, "pass", f"skipped: fewer than {LENGTH_MIN_SAMPLES} samples", {"n": n})
    if center is None or not scale:
        return _result(name, "pass", "skipped: no length spread in dataset", {"n": n})
    z = (math.log1p(max(tokens_est, 0)) - center) / scale
    details = {
        "tokens_est": int(tokens_est),
        "z": round(z, 2),
        "dataset_median_tokens": round(median_tokens, 1) if median_tokens is not None else None,
        "method": method,
    }
    direction = "long" if z > 0 else "short"
    med = f"{median_tokens:.0f}" if median_tokens is not None else "?"
    msg = f"{tokens_est} tokens vs dataset median {med} (robust z {z:+.1f})"
    if abs(z) > LENGTH_FAIL_Z:
        return _result(name, "fail", f"extreme {direction} outlier: {msg}", details)
    if abs(z) > LENGTH_WARN_Z:
        return _result(name, "warn", f"{direction} outlier: {msg}", details)
    return _result(name, "pass", msg, details)


# ---------------------------------------------------------------------------------------------
# exact_duplicate / near_duplicate
# ---------------------------------------------------------------------------------------------


def check_exact_duplicate(duplicate_of: Sequence[int], same_prompt_as: Sequence[int] = ()) -> CheckResult:
    """`duplicate_of`: earlier samples with the same content hash; `same_prompt_as`: other samples
    with the same normalised prompt+context but a different response."""
    name = "exact_duplicate"
    if duplicate_of:
        details: dict[str, Any] = {"duplicate_of": list(duplicate_of)}
        if same_prompt_as:
            details["same_prompt_as"] = list(same_prompt_as)
        return _result(name, "fail", f"exact duplicate of {len(duplicate_of)} earlier sample(s), first is sample id {duplicate_of[0]}", details)
    if same_prompt_as:
        return _result(
            name,
            "warn",
            f"same prompt appears in {len(same_prompt_as)} other sample(s) with a different response",
            {"same_prompt_as": list(same_prompt_as)},
        )
    return _result(name, "pass", "no duplicate", {})


def check_near_duplicate(
    similar_earlier: Sequence[tuple[int, float]],
    similar_later: Sequence[tuple[int, float]] = (),
    is_exact_duplicate: bool = False,
) -> CheckResult:
    """Pairs are (sample id, jaccard). Only similarity to an *earlier* sample is a defect; the
    first occurrence passes but still lists its near copies in details."""
    name = "near_duplicate"
    earlier = sorted(similar_earlier, key=lambda p: -p[1])[:SIMILAR_TOP_K]
    later = sorted(similar_later, key=lambda p: -p[1])[:SIMILAR_TOP_K]
    if earlier and earlier[0][1] >= NEAR_DUP_WARN_JACCARD:
        best = earlier[0][1]
        details = {"similar_to": [{"id": i, "jaccard": j} for i, j in earlier]}
        msg = f"near duplicate of sample id {earlier[0][0]} (Jaccard {best:.2f}); {len(earlier)} similar earlier sample(s)"
        severity = "fail" if best >= NEAR_DUP_FAIL_JACCARD and not is_exact_duplicate else "warn"
        return _result(name, severity, msg, details)
    if later:
        return _result(
            name,
            "pass",
            f"first occurrence; {len(later)} later near copy(ies)",
            {"similar_to": [{"id": i, "jaccard": j} for i, j in later]},
        )
    return _result(name, "pass", "no near duplicate", {})


# ---------------------------------------------------------------------------------------------
# pii
# ---------------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.]\d|\w)")
# 13-19 digits, optionally grouped by single spaces/dashes (Luhn-checked afterwards).
_CARD_RE = re.compile(r"(?<![\d\w])\d(?:[ -]?\d){12,18}(?![\d\w])")
# Aadhaar: 12 digits, first 2-9, written 4-4-4 with spaces/dashes.
_AADHAAR_RE = re.compile(r"(?<![\d\w])(?<!\d[ -])[2-9]\d{3}([ -])\d{4}\1\d{4}(?![\d\w])(?![ -]\d)")
_PHONE_RES = [
    # Indian mobile: optional +91 / 0 prefix, 10 digits starting 6-9 (optionally 5+5 split).
    re.compile(r"(?<![\w+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\w])"),
    # International with explicit +country code.
    re.compile(r"(?<![\w+])\+(?!91)[1-9]\d{0,2}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?![\w])"),
    # North-American style (555) 123-4567 / 555-123-4567 / 555.123.4567.
    re.compile(r"(?<![\w(])(?:\(\d{3}\)\s?|\d{3}[.-])\d{3}[.-]\d{4}(?![\w])"),
]
_SECRET_RES = [
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic style keys
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),  # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),  # Hugging Face token
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
]
_VERSION_HINT_RE = re.compile(r"(?:version|release|ver\.?|v|build|firmware|rev\.?)\s*$")
_SECRET_HINT_RE = re.compile(r"sk-|AKIA|ASIA|gh[pousr]_|xox[abprs]-|AIza|hf_|-----BEGIN")
# Priority when spans overlap (higher wins).
_PII_PRIORITY = {"secret": 6, "credit_card": 5, "aadhaar": 4, "email": 3, "phone": 2, "ip_address": 1}


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pii(text: str, field_name: str) -> list[dict[str, Any]]:
    """PII spans {field, start, end, kind} with offsets into `text`."""
    if not text:
        return []
    cands: list[tuple[int, int, str]] = []
    if _SECRET_HINT_RE.search(text):
        for rx in _SECRET_RES:
            cands += [(m.start(), m.end(), "secret") for m in rx.finditer(text)]
    if "@" in text:
        for m in _EMAIL_RE.finditer(text):
            cands.append((m.start(), m.end(), "email"))
    if sum(map(text.count, "0123456789")) < 4:  # every numeric pattern needs at least 4 digits
        cands.sort(key=lambda c: c[0])
        return [{"field": field_name, "start": s, "end": e, "kind": k} for s, e, k in cands]
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and len(set(digits)) >= 2 and _luhn_ok(digits):
            cands.append((m.start(), m.end(), "credit_card"))
    for m in _AADHAAR_RE.finditer(text):
        cands.append((m.start(), m.end(), "aadhaar"))
    for rx in _PHONE_RES:
        for m in rx.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 10 <= len(digits) <= 15 and len(set(digits)) > 2:
                cands.append((m.start(), m.end(), "phone"))
    for m in _IPV4_RE.finditer(text):
        octets = m.group().split(".")
        before = text[max(0, m.start() - 12) : m.start()].lower()
        if octets[0] == "0" or _VERSION_HINT_RE.search(before):  # "0.x.y.z" / "version 1.2.3.4" are not hosts
            continue
        cands.append((m.start(), m.end(), "ip_address"))
    # Resolve overlaps: highest priority first, then longest.
    cands.sort(key=lambda c: (-_PII_PRIORITY[c[2]], -(c[1] - c[0]), c[0]))
    taken: list[tuple[int, int]] = []
    spans = []
    for s, e, kind in cands:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        spans.append({"field": field_name, "start": s, "end": e, "kind": kind})
    spans.sort(key=lambda d: d["start"])
    return spans


def check_pii(prompt: str | None, context: str | None, response: str | None) -> CheckResult:
    name = "pii"
    spans = find_pii(prompt or "", "prompt") + find_pii(context or "", "context") + find_pii(response or "", "response")
    if not spans:
        return _result(name, "pass", "no PII found", {"spans": []})
    kinds: dict[str, int] = {}
    for s in spans:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    summary = ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items()))
    details = {"spans": spans, "kinds": kinds}
    if len(spans) >= PII_FAIL_HITS or PII_HIGH_RISK & kinds.keys():
        return _result(name, "fail", f"{len(spans)} PII hit(s): {summary}", details)
    return _result(name, "warn", f"{len(spans)} PII hit(s): {summary}", details)


# ---------------------------------------------------------------------------------------------
# non_english
# ---------------------------------------------------------------------------------------------

_EN_STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but by can could did do does for from had has
    have he her his how i if in into is it its just like many may me more most my no not of on one only or other
    our out over she so some such than that the their them then there these they this those to up was we were what
    when where which while who will with would you your""".split()
)
# Common function words of other Latin-script languages (es, fr, de, pt, it, nl, id) that are not English words.
_FOREIGN_STOPWORDS = frozenset(
    """el la los las del que en por para con una unos es son est le les des du une et dans pour pas qui sur avec au
    aux der die das und ist nicht ein eine mit den dem zu auf von sich auch wie o os um uma não com da do dos das
    em il di che per non sono della het een niet op voor zijn yang dan dengan untuk tidak ini itu ada di""".split()
)
_CODE_FENCE_RE = re.compile(r"```")
_FENCED_ANY_RE = re.compile(r"```.*?(?:```|$)", re.S)
_CODE_SYMBOLS = "{}()[];=<>#$_\\|&*/+`"
_LETTER_RE = re.compile(r"[^\W\d_]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"\s")


def _code_like(text: str) -> bool:
    if _CODE_FENCE_RE.search(text):
        return True
    non_space = len(text) - len(_SPACE_RE.findall(text))
    if non_space < 20:
        return False
    return sum(map(text.count, _CODE_SYMBOLS)) / non_space >= CODE_SYMBOL_DENSITY


def detect_language(text: str) -> tuple[str, dict[str, Any]]:
    """Heuristic: returns ("en" | "other", details)."""
    n_letters = len(_LETTER_RE.findall(text))
    ascii_letters = len(_ASCII_LETTER_RE.findall(text))
    ascii_share = ascii_letters / n_letters if n_letters else 1.0
    words = _WORD_RE.findall(text.lower())
    en = sum(w in _EN_STOPWORDS for w in words)
    fo = sum(w in _FOREIGN_STOPWORDS for w in words)
    en_ratio = en / len(words) if words else 0.0
    fo_ratio = fo / len(words) if words else 0.0
    details: dict[str, Any] = {
        "letters": n_letters,
        "ascii_share": round(ascii_share, 3),
        "en_stopword_ratio": round(en_ratio, 3),
        "foreign_stopword_ratio": round(fo_ratio, 3),
    }
    if n_letters < LANG_MIN_LETTERS:
        details["reason"] = "too short"
        return "en", details
    if ascii_share < LANG_MIN_ASCII_SHARE:
        details["reason"] = "non-Latin script"
        return "other", details
    if (
        len(words) >= LANG_MIN_WORDS_FOR_STOPWORDS
        and fo_ratio > LANG_MAX_FOREIGN_RATIO
        and fo_ratio > en_ratio
        and en_ratio < LANG_MAX_EN_RATIO_FOR_FOREIGN
    ):
        details["reason"] = "foreign function words"
        return "other", details
    return "en", details


def check_non_english(prompt: str | None, context: str | None, response: str | None) -> CheckResult:
    """details.lang is "en" or "other" (stored as Sample.lang by the job)."""
    name = "non_english"
    # Judge prompt + response; context is often quoted source material (names, places, foreign terms).
    # Fenced code is removed first; text that is still code-like (symbol density) is not judged.
    text = _FENCED_ANY_RE.sub(" ", f"{prompt or ''}\n{response or ''}")
    if _code_like(text):
        prompt_text = _FENCED_ANY_RE.sub(" ", prompt or "")
        if _code_like(prompt_text) or sum(c.isalpha() for c in prompt_text) < LANG_MIN_LETTERS:
            return _result(name, "pass", "skipped: mostly code", {"lang": "en", "code": True})
        text = prompt_text
    lang, details = detect_language(text)
    details["lang"] = lang
    if lang == "other":
        return _result(
            name,
            "warn",
            f"likely non-English ({details.get('reason')}): ASCII letters {details['ascii_share']:.0%}, "
            f"English stopwords {details['en_stopword_ratio']:.0%}",
            details,
        )
    return _result(name, "pass", f"English (ASCII letters {details['ascii_share']:.0%})", details)


# ---------------------------------------------------------------------------------------------
# refusal_boilerplate
# ---------------------------------------------------------------------------------------------

_REFUSAL_PHRASES = [
    r"as an ai(?: language)? model",
    r"as a large language model",
    r"as an ai assistant",
    r"i(?:'m| am) (?:just )?an ai\b",
    r"i(?:'m| am) sorry,? but i (?:can(?:'|no)t|am unable|am not able|won't)",
    r"i cannot (?:assist|help|provide|fulfill|comply)",
    r"i can(?:'|no)t (?:assist|help) with (?:that|this)",
    r"i(?:'m| am) (?:unable|not able) to (?:assist|help|provide|comply|fulfill)",
    r"i don(?:'|’)?t have personal (?:opinions|beliefs|feelings|experiences)",
    r"i do not have personal (?:opinions|beliefs|feelings|experiences)",
    r"my knowledge cutoff",
    r"(?:trained|developed|created) by openai",
]
_REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PHRASES), re.I)
# Cheap substring pre-filter: every refusal phrase contains one of these.
_REFUSAL_ANCHORS = ("as an ai", "as a large", "an ai", "sorry", "i cannot", "i can't", "i cant", "unable", "not able",
                    "personal", "knowledge cutoff", "openai")
_VENDOR_RE = re.compile(r"\b(?:openai|chatgpt)\b", re.I)
_VENDOR_SELF_RE = re.compile(
    r"\b(?:I|I'm|me|my)\b[^.\n]{0,60}\b(?:openai|chatgpt)\b|\b(?:openai|chatgpt)\b[^.\n]{0,60}\b(?:I|I'm|me|my)\b", re.I
)


def check_refusal_boilerplate(prompt: str | None, response: str | None, context: str | None = None) -> CheckResult:
    name = "refusal_boilerplate"
    resp = (response or "").replace("’", "'")
    low = resp.lower()
    hits: list[str] = []
    if any(a in low for a in _REFUSAL_ANCHORS):
        hits = sorted({m.group(0).lower() for m in _REFUSAL_RE.finditer(resp)})
    # OpenAI/ChatGPT only counts as boilerplate when the response talks about itself near it
    # ("I am ChatGPT", "my training by OpenAI"); encyclopedic mentions (GPT-4 in Copilot) are fine.
    vendor = [m.group(0) for m in _VENDOR_SELF_RE.finditer(resp)] if ("openai" in low or "chatgpt" in low) else []
    asked = f"{prompt or ''} {context or ''}"
    if vendor and _VENDOR_RE.search(asked):
        vendor = []  # the prompt is about OpenAI/ChatGPT: mentioning it is not boilerplate
    if not hits and not vendor:
        return _result(name, "pass", "no refusal/boilerplate phrases", {})
    nw = len(resp.split())
    details: dict[str, Any] = {"phrases": hits, "vendor_mentions": len(vendor), "response_words": nw}
    if hits and nw <= REFUSAL_FAIL_MAX_WORDS:
        return _result(name, "fail", f"response is mostly a refusal ({nw} words): '{hits[0]}'", details)
    if hits:
        return _result(name, "warn", f"{len(hits)} refusal/AI-boilerplate phrase(s): '{hits[0]}'", details)
    return _result(name, "warn", f"response refers to itself as OpenAI/ChatGPT {len(vendor)} time(s) unprompted", details)


# ---------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------

_REPEAT_RE = re.compile(r"(\S)\1{%d,}" % (REPEATED_CHAR_RUN - 1))
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_PAIRS = {")": "(", "]": "[", "}": "{"}
_END_OK = set(".!?。！？…:;)]}\"'”’`*>|-_~%$")


_DANGLING_WORDS = frozenset(
    """the a an and or but nor of to in on at for with by from as that which who whose is are was were be been their
    his her its our your my this these those into than because although while if when where whether such""".split()
)


def _brackets_balanced(code: str) -> bool:
    # strip string literals and comments so quoted brackets don't count
    code = re.sub(r"(\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')", "", code)
    code = re.sub(r"(#|//)[^\n]*", "", code)
    stack: list[str] = []
    for ch in code:
        if ch in "([{":
            stack.append(ch)
        elif ch in _PAIRS:
            if not stack or stack.pop() != _PAIRS[ch]:
                return False
    return not stack


def _trigrams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def check_formatting(prompt: str | None, response: str | None) -> CheckResult:
    name = "formatting"
    resp = response or ""
    issues: list[str] = []
    details: dict[str, Any] = {}

    if resp.count("```") % 2 == 1:
        issues.append("unbalanced_code_fence")
    blocks = _FENCED_BLOCK_RE.findall(resp)
    if any(not _brackets_balanced(b) for b in blocks):
        issues.append("unbalanced_brackets")

    m = _REPEAT_RE.search(resp)
    if m:
        issues.append("repeated_characters")
        details["repeated_char"] = m.group(1)
        details["repeated_run"] = len(m.group(0))

    rw = _words(resp.lower())
    pw = _words((prompt or "").lower())
    if len(rw) >= PROMPT_ECHO_MIN_WORDS and pw:
        # Echo = the response is (almost) the prompt itself, in both directions. One-directional
        # containment flags legit answers that restate the question or reorder a list from the prompt.
        rt, pt = _trigrams(rw), _trigrams(pw)
        common = len(rt & pt)
        containment = common / len(rt) if rt else 0.0
        coverage = common / len(pt) if pt else 0.0
        if containment >= PROMPT_ECHO_CONTAINMENT and coverage >= PROMPT_ECHO_COVERAGE:
            issues.append("prompt_echo")
            details["prompt_overlap"] = round(containment, 3)
            details["prompt_coverage"] = round(coverage, 3)

    stripped = resp.rstrip()
    if len(rw) >= TRUNCATION_MIN_WORDS and stripped and "```" not in resp:
        last_line = stripped.splitlines()[-1].strip()
        is_list_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s", last_line))
        # Human writers often drop the final full stop, so a missing period alone is not truncation:
        # require a dangling function word ("... imposed penalties on") or an unclosed bracket ("[3").
        last_word = rw[-1] if rw else ""
        dangling = last_word in _DANGLING_WORDS or last_line.count("[") > last_line.count("]") or (
            last_line.count("(") > last_line.count(")")
        )
        if stripped[-1] not in _END_OK and stripped[-1].isalnum() and not is_list_item and dangling:
            issues.append("truncated")
            details["ends_with"] = stripped[-40:]

    details["issues"] = issues
    if not issues:
        return _result(name, "pass", "no formatting issues", details)
    severity = "fail" if len(issues) > 1 else "warn"
    return _result(name, severity, f"{len(issues)} formatting issue(s): {', '.join(issues)}", details)


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------

_SEVERITY_RANK = {"pass": 0, "warn": 1, "fail": 2}


def summarize(results: Sequence[CheckResult]) -> tuple[str, float | None]:
    """(qc_status, qc_score): worst severity, share of checks passed."""
    if not results:
        return "pending", None
    worst = max(results, key=lambda r: _SEVERITY_RANK.get(r.severity, 2)).severity
    score = sum(1 for r in results if r.passed) / len(results)
    return worst, round(score, 4)


def run_all(row: SampleRow, stats: DatasetStats) -> list[CheckResult]:
    """All 8 checks for one sample, using dataset-wide stats. Raises on unexpected errors."""
    idx = stats.index_of.get(row.id, row.sample_index)
    duplicate_of = stats.duplicate_of.get(row.id, [])
    near = stats.near_dups.get(row.id, [])  # later exact copies are not in the near-dup search at all
    earlier = [(o, j) for o, j in near if stats.index_of.get(o, -1) < idx]
    later = [(o, j) for o, j in near if stats.index_of.get(o, -1) >= idx]
    checks: list[Callable[[], CheckResult]] = [
        lambda: check_empty_or_short(row.prompt, row.response),
        lambda: check_length_outlier(
            row.tokens_est, stats.token_center, stats.token_scale, stats.n, stats.token_median, stats.token_scale_method
        ),
        lambda: check_exact_duplicate(duplicate_of, stats.prompt_conflicts.get(row.id, [])),
        lambda: check_near_duplicate(earlier, later, is_exact_duplicate=bool(duplicate_of)),
        lambda: check_pii(row.prompt, row.context, row.response),
        lambda: check_non_english(row.prompt, row.context, row.response),
        lambda: check_refusal_boilerplate(row.prompt, row.response, row.context),
        lambda: check_formatting(row.prompt, row.response),
    ]
    return [c() for c in checks]

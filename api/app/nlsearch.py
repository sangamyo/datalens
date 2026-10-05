"""Natural-language sample search: turn a free-text query into a validated SampleFilter.

Two parsers:
- `parse_rules`: deterministic regex/keyword parser, always available.
- `parse_llm`: asks Gemini or Claude for JSON matching SampleFilter; used only when an API key is
  configured, and returns None on any error so the caller falls back to the rules.

Only the query text is ever sent to an LLM, never dataset contents. The output is always a
Pydantic-validated SampleFilter (never SQL).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from typing import Literal

import httpx
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import CHECK_NAMES, SampleFilter

log = logging.getLogger(__name__)

ParserName = Literal["llm", "rules"]
STATUS_ORDER = ("pending", "pass", "warn", "fail")
LLM_TIMEOUT_S = 10.0

SHORT_ANSWER_MAX_TOKENS = 40
LONG_SAMPLE_MIN_TOKENS = 400
WORDS_TO_TOKENS = 1.3
CHARS_PER_TOKEN = 4
APPROX_TOLERANCE = 0.2  # "about 200 tokens" -> 160..240

# Canonical category names (databricks-dolly-15k + HuggingFaceH4/no_robots, normalised).
CATEGORIES = (
    "open_qa",
    "closed_qa",
    "general_qa",
    "classification",
    "brainstorming",
    "information_extraction",
    "summarization",
    "creative_writing",
    "generation",
    "rewrite",
    "chat",
    "coding",
)

# How each canonical category may be spelled in a dataset's own `category` column.
CATEGORY_VARIANTS: dict[str, tuple[str, ...]] = {
    "open_qa": ("open_qa", "open qa", "open-qa", "openqa"),
    "closed_qa": ("closed_qa", "closed qa", "closed-qa", "closedqa"),
    "general_qa": ("general_qa", "general qa", "general-qa"),
    "classification": ("classification", "classify"),
    "brainstorming": ("brainstorming", "brainstorm"),
    "information_extraction": ("information_extraction", "information extraction", "extract", "extraction"),
    "summarization": ("summarization", "summarisation", "summarize", "summarise", "summary"),
    "creative_writing": ("creative_writing", "creative writing", "creative"),
    "generation": ("generation", "generate"),
    "rewrite": ("rewrite", "rewriting"),
    "chat": ("chat", "conversation"),
    "coding": ("coding", "code", "programming"),
}


def _norm_cat(s: str) -> str:
    return re.sub(r"[\s_-]+", " ", s.strip().lower())


def resolve_category(category: str, available: Iterable[str]) -> str:
    """Map a canonical category (e.g. "brainstorming") to the spelling a dataset actually uses
    (e.g. no_robots' "Brainstorm"). Returns `category` unchanged when nothing matches."""
    available = [a for a in available if a]
    want = _norm_cat(category)
    for a in available:
        if _norm_cat(a) == want:
            return a
    variants = {_norm_cat(v) for v in CATEGORY_VARIANTS.get(want.replace(" ", "_"), ())}
    for a in available:
        if _norm_cat(a) in variants:
            return a
    return category


# ---------------------------------------------------------------------------------------------
# Rule-based parser
# ---------------------------------------------------------------------------------------------

_W = r"(?<![\w-])"  # left word boundary that also treats '-' as part of a word
_E = r"(?![\w-])"

_NUM = r"(\d+(?:[.,]\d+)?)\s*(k)?"
_UNIT = r"(tokens?|toks?|words?|characters?|chars?)"

_MIN_CMP = (
    r"longer\s+than|more\s+than|greater\s+than|bigger\s+than|larger\s+than|over|above|exceeding|exceeds|"
    r"at\s+least|no\s+less\s+than|no\s+fewer\s+than|no\s+shorter\s+than|minimum(?:\s+of)?|min\.?|>=|=>|>"
)
_MAX_CMP = (
    r"shorter\s+than|less\s+than|fewer\s+than|under|below|within|at\s+most|no\s+more\s+than|"
    r"no\s+longer\s+than|up\s+to|maximum(?:\s+of)?|max\.?|<=|=<|<"
)
_LENGTH_CMP = r"longer\s+than|shorter\s+than|no\s+longer\s+than|no\s+shorter\s+than"

_BETWEEN_RE = re.compile(
    rf"(?:between|from)\s+{_NUM}\s*{_UNIT}?\s*(?:and|to|-)\s*{_NUM}\s*{_UNIT}(?![a-z])"
    rf"|(?<![\w.]){_NUM}\s*{_UNIT}?\s*(?:-|to)\s*{_NUM}\s*{_UNIT}(?![a-z])",
    re.I,
)
_CMP_RE = re.compile(rf"(?<![a-z])({_MIN_CMP}|{_MAX_CMP})\s*{_NUM}\s*{_UNIT}?(?![a-z])", re.I)
_APPROX_RE = re.compile(
    rf"(?<![a-z])(?:about|around|roughly|approximately|approx\.?|circa|~)\s*{_NUM}\s*{_UNIT}(?![a-z])", re.I
)
_PLUS_RE = re.compile(rf"(?<![\w.]){_NUM}\s*\+\s*{_UNIT}(?![a-z])", re.I)  # "500+ tokens"
_POSTFIX_RE = re.compile(
    rf"(?<![\w.]){_NUM}\s*{_UNIT}\s*(\+|or\s+(?:more|longer|above|over|greater)|plus"
    rf"|or\s+(?:less|fewer|shorter|under|below))(?![a-z])",
    re.I,
)
_MIN_CMP_RE = re.compile(rf"^(?:{_MIN_CMP})$", re.I)
_LENGTH_CMP_RE = re.compile(rf"^(?:{_LENGTH_CMP})$", re.I)

_SHORT_RE = re.compile(
    r"\b(?:short|brief|terse|concise)(?=\s+(?:[a-z_-]+\s+){0,2}?(?:answers?|responses?|replies|reply|outputs?|completions?|samples?|"
    r"examples?|ones|rows?|entries|items?)\b)",
    re.I,
)
_LONG_RE = re.compile(
    r"\b(?:long|lengthy|verbose)(?=\s+(?:[a-z_-]+\s+){0,2}?(?:answers?|responses?|replies|reply|outputs?|completions?|samples?|"
    r"examples?|ones|rows?|entries|items?|prompts?|instructions?|texts?)\b)",
    re.I,
)

# A check phrase may be wrapped in "failed the ... check" / "... issues": that wrapping belongs to
# the check, not to the qc_status.
_CHECK_PREFIX = (
    r"(?:\bfail(?:s|ed|ing)?\s+(?:(?:the|on|for|by|at)\s+)+"
    r"|\b(?:flagged|bad|poor|has|have|having|with|contain(?:s|ing)?|show(?:s|ing)?)"
    r"\s+(?:(?:the|on|for|by|as|an?|any|some)\s+)*)?"
)
_CHECK_SUFFIX = r"(?:\s+(?:issues?|problems?|errors?|warnings?|flags?|checks?|failures?|detected|found|hits?))*"

# (pattern, check_name) in priority order: more specific phrases first; matched spans are
# consumed so e.g. "repeated characters" never also counts as "repeated" (exact_duplicate).
_CHECK_PATTERNS: list[tuple[str, str]] = [
    (r"near[_\s-]*dup(?:e|es|licates?|licated)?s?|nearly\s+(?:identical|the\s+same)|almost\s+(?:identical|the\s+same)|"
     r"(?:very\s+)?similar(?:\s+(?:samples?|prompts?|ones|examples?))?|paraphras(?:ed|es)\s+dup(?:e|es|licates?)s?|"
     r"fuzzy\s+dup(?:e|es|licates?)s?|semantic\s+dup(?:e|es|licates?)s?|paraphrased|"
     r"(?:almost|near|nearly)\s+duplicated?",
     "near_duplicate"),
    (r"formatting|format(?:ted)?\s+(?:badly|poorly|wrong(?:ly)?)|badly\s+formatted|poorly\s+formatted|mis-?formatted|"
     r"malformed|broken\s+(?:code|markdown|formatting|json|html)|unclosed\s+(?:code\s+)?(?:blocks?|fences?)|"
     r"unbalanced\s+(?:code\s+)?(?:blocks?|fences?|brackets?)|truncat(?:ed|ion)|cut\s+off|cut-off|"
     r"incomplete(?:\s+(?:answers?|responses?|sentences?))?|repeated\s+(?:characters?|chars?|letters?|symbols?|punctuation)|"
     r"repetitive(?:\s+(?:characters?|text))?|garbled|weird\s+characters?|mojibake",
     "formatting"),
    (r"exact[_\s-]*dup(?:e|es|licates?|licated)?s?|dup(?:e|es)|dup(?:licate|licates|licated|lication)s?|"
     r"repeated(?:\s+(?:samples?|prompts?|rows?|entries|examples?))?|repeats|copies|copied|identical(?:\s+(?:samples?|prompts?|rows?))?",
     "exact_duplicate"),
    (r"pii|personal(?:ly)?\s+(?:data|info(?:rmation)?|details|identifiable(?:\s+information)?)|private\s+(?:data|info(?:rmation)?)|"
     r"e-?mails?(?:\s+addresses?)?|email\s+addresses|phone(?:\s+numbers?)?|telephone(?:\s+numbers?)?|"
     r"sensitive(?:\s+(?:data|info(?:rmation)?|content))?|secrets?|api\s+keys?|passwords?|credentials?|"
     r"credit[\s-]*cards?(?:\s+numbers?)?|ssns?|social\s+security(?:\s+numbers?)?|ip\s+addresses?|addresses",
     "pii"),
    (r"non[\s_-]*english|not\s+(?:in\s+)?english|foreign(?:\s+languages?)?|other\s+languages?|another\s+language|"
     r"different\s+languages?|multilingual|hindi|spanish|french|german|chinese|japanese|korean|arabic|russian|"
     r"portuguese|italian|dutch|turkish|vietnamese|indonesian|bengali|urdu|tamil|telugu|marathi",
     "non_english"),
    (r"refus(?:als?|ing|es|ed|e)|refusal[_\s-]*boilerplate|as\s+an\s+ai(?:\s+(?:language\s+)?model)?|"
     r"(?:ai\s+)?boilerplate|canned(?:\s+(?:answers?|responses?|replies))?|i\s+can(?:'|no)?t\s+help|i\s+cannot|"
     r"declin(?:es|ed|ing)|disclaimers?|apologi[sz](?:es|ing|e)|sorry",
     "refusal_boilerplate"),
    (r"empty(?:[_\s-]*or[_\s-]*short)?|blank|too\s+short|very\s+short|extremely\s+short|one[\s-]+word(?:\s+(?:answers?|responses?))?|"
     r"single[\s-]+word(?:\s+(?:answers?|responses?))?|missing\s+(?:answers?|responses?)|no\s+(?:answers?|responses?)|"
     r"trivial(?:\s+(?:answers?|responses?))?",
     "empty_or_short"),
    (r"length[_\s-]*outliers?|outliers?|too\s+long|very\s+long|extremely\s+long|overly\s+long|"
     r"unusual(?:ly)?\s+(?:length|long|sized?)|abnormal(?:ly)?\s+(?:length|long)|extreme\s+lengths?|weird\s+lengths?|"
     r"odd\s+lengths?",
     "length_outlier"),
]
_CHECK_RES = [
    (re.compile(rf"{_CHECK_PREFIX}{_W}(?:{p}){_E}{_CHECK_SUFFIX}", re.I), name) for p, name in _CHECK_PATTERNS
]

_STATUS_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    # Negations / specific phrases first; their spans are consumed before the generic words run.
    (r"\b(?:not|never|didn'?t|did\s+not|haven'?t|hasn'?t|have\s+not|has\s+not)\s+(?:been\s+|yet\s+)?"
     r"(?:qc'?d|checked|scored|evaluated|analy[sz]ed|inspected|reviewed|run)(?:\s+yet)?\b", ("pending",)),
    (r"\b(?:not|didn'?t|did\s+not|doesn'?t|does\s+not|don'?t)\s+pass(?:ed|ing)?\b|\bnon[\s-]?passing\b", ("warn", "fail")),
    (r"\b(?:no|without|zero)\s+(?:problems?|issues?|warnings?|errors?|failures?|flags?)\b", ("pass",)),
    (r"\b(?:only\s+warn(?:ings?|s)?|warn(?:ings?|s)?\s+only|just\s+warn(?:ings?|s)?)\b", ("warn",)),
    (r"\bneeds?\s+(?:a\s+)?(?:review|attention|fixing|cleaning|cleanup)\b|\bto\s+review\b", ("warn", "fail")),
    (r"\b(?:pending|unchecked|unscored|unprocessed|unreviewed|awaiting\s+qc|qc\s+pending|in\s+progress)\b", ("pending",)),
    (r"\b(?:high|good|top)[-\s]quality\b", ("pass",)),
    (r"\b(?:low|poor|bad)[-\s]quality\b", ("warn", "fail")),
    (r"\b(?:warn(?:ings?|s|ed)?|problems?|problematic|issues?|flagged|suspicious|questionable|errors?|"
     r"anomal(?:y|ies|ous)|noisy|dirty)\b", ("warn", "fail")),
    (r"\b(?:fail(?:s|ed|ing|ures?)?|broken|bad|worst|garbage|junk)\b", ("fail",)),
    (r"\b(?:pass(?:es|ed|ing)?|clean|good|ok|okay|fine|valid|perfect)\b", ("pass",)),
]
_STATUS_RES = [(re.compile(p, re.I), s) for p, s in _STATUS_PATTERNS]

# (pattern, canonical category)
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    (r"open[\s_-]*qa|open[\s-]+ended(?:\s+(?:questions?|qa))?|open\s+questions?", "open_qa"),
    (r"closed[\s_-]*qa|closed[\s-]+book(?:\s+(?:questions?|qa))?|reading\s+comprehension|closed\s+questions?", "closed_qa"),
    (r"general[\s_-]*qa|general\s+(?:knowledge\s+)?questions?", "general_qa"),
    (r"information[\s_-]+extraction|extraction|extract(?:ing|s)?", "information_extraction"),
    (r"classification|classif(?:y|ying|ied)|categori[sz]ation", "classification"),
    (r"brainstorm(?:ing|s)?|ideas?", "brainstorming"),
    (r"summar(?:y|ies|i[sz]ation|i[sz]e|i[sz]ing|i[sz]ations)", "summarization"),
    (r"creative[\s_-]*writing|creative|stor(?:y|ies)|poems?|poetry|fiction|haikus?|songs?|lyrics", "creative_writing"),
    (r"generation|generat(?:e|ing)", "generation"),
    (r"rewrit(?:e|es|ing|ten)|rephras(?:e|ing)", "rewrite"),
    (r"chats?|conversations?|conversational|dialog(?:ue)?s?|multi[\s-]?turn", "chat"),
    (r"coding|code|programming|programs?", "coding"),
]
_CATEGORY_RES = [(re.compile(rf"{_W}(?:{p}){_E}", re.I), c) for p, c in _CATEGORY_PATTERNS]

_ENGLISH_RE = re.compile(r"(?<![\w-])(?:in\s+)?english(?:[\s-]+only)?(?![\w-])|\bonly\s+english\b", re.I)

_QUOTE_RE = re.compile(r"\"([^\"]+)\"|“([^”]+)”|'([^']{2,})'(?!\w)|`([^`]+)`")
_TEXT_TRIGGER = (
    r"mention(?:s|ing|ed)?|contain(?:s|ing)?|includ(?:es|ing)|referenc(?:es|ing)|"
    r"(?:that|which|who)\s+(?:talk|talks|are|is)\s+about|talk(?:s|ing)?\s+about|about|regarding|"
    r"related\s+to|on\s+the\s+topic\s+of|with\s+the\s+(?:word|phrase|term|text)|with\s+(?:word|phrase|term|text)|"
    r"using\s+the\s+(?:word|phrase|term)|that\s+say|saying|says|matching|with\s+keyword"
)
_TEXT_RE = re.compile(rf"\b(?:{_TEXT_TRIGGER})\s+(?:the\s+(?:word|phrase|term|topic)\s+)?(?P<x>.+)$", re.I)
_TEXT_STOP = re.compile(
    r"\s+(?:in|with|from|that|which|where|and|or|but|having|whose|longer|shorter|over|under|above|below|"
    r"more|less|fewer|between|at|than|for|of\s+category|category|status|tokens?|words?|marked|flagged|failed|"
    r"failing|passing|passed|only|written|not)\b.*$",
    re.I,
)
_LEADING_JUNK = re.compile(r"^(?:the|a|an|some|any)\s+", re.I)
_FILLER_WORDS = {
    "samples", "sample", "examples", "example", "rows", "row", "entries", "entry", "items", "item", "ones", "data",
    "show", "find", "get", "list", "give", "me", "all", "the", "a", "an", "any", "some", "with", "that", "which",
    "are", "is", "in", "of", "and", "or", "please", "only", "answers", "answer", "responses", "response", "prompts",
    "prompt", "instructions", "instruction", "dataset", "records", "record", "things", "stuff", "everything",
    "it", "them", "those", "these", "this", "things", "qc", "check", "checks", "status", "category",
}


class _Text:
    """Lower-cased working copy of the query whose matched spans get blanked out."""

    def __init__(self, s: str) -> None:
        self.s = s

    def consume(self, m: re.Match[str]) -> None:
        a, b = m.span()
        self.s = self.s[:a] + " " * (b - a) + self.s[b:]


def _num(value: str, k: str | None) -> float:
    n = float(value.replace(",", "")) if value.count(",") == 1 and len(value.split(",")[1]) == 3 else float(
        value.replace(",", ".")
    )
    return n * 1000 if k else n


def _to_tokens(n: float, unit: str | None) -> int:
    u = (unit or "tokens").lower()
    if u.startswith("word"):
        return max(0, round(n * WORDS_TO_TOKENS))
    if u.startswith("char"):
        return max(0, round(n / CHARS_PER_TOKEN))
    return max(0, round(n))


def _parse_tokens(t: _Text) -> tuple[int | None, int | None]:
    lo: int | None = None
    hi: int | None = None

    for m in list(_BETWEEN_RE.finditer(t.s)):
        g = m.groups()
        if g[0] is not None:
            a, ak, au, b, bk, bu = g[0:6]
        else:
            a, ak, au, b, bk, bu = g[6:12]
        unit = bu or au
        x, y = sorted((_to_tokens(_num(a, ak), au or unit), _to_tokens(_num(b, bk), unit)))
        lo, hi = x, y
        t.consume(m)

    for m in list(_APPROX_RE.finditer(t.s)):
        center = _to_tokens(_num(m.group(1), m.group(2)), m.group(3))
        lo, hi = math.floor(center * (1 - APPROX_TOLERANCE)), math.ceil(center * (1 + APPROX_TOLERANCE))
        t.consume(m)

    for m in list(_PLUS_RE.finditer(t.s)):
        lo = _to_tokens(_num(m.group(1), m.group(2)), m.group(3))
        t.consume(m)

    for m in list(_POSTFIX_RE.finditer(t.s)):
        n = _to_tokens(_num(m.group(1), m.group(2)), m.group(3))
        if re.search(r"less|fewer|shorter|under|below", m.group(4), re.I):
            hi = n
        else:
            lo = n
        t.consume(m)

    for m in list(_CMP_RE.finditer(t.s)):
        cmp_, value, k, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        cmp_norm = re.sub(r"\s+", " ", cmp_.strip().lower())
        if unit is None and not _LENGTH_CMP_RE.match(cmp_norm):
            # a bare number ("over 3") only counts as a length after "longer/shorter than" or with
            # a "1k" suffix; otherwise it may mean something else entirely
            if not k:
                continue
        n = _to_tokens(_num(value, k), unit)
        strict = cmp_norm in {"longer than", "more than", "greater than", "bigger than", "larger than", "over",
                              "above", "exceeding", "exceeds", ">", "shorter than", "less than", "fewer than",
                              "under", "below", "<"}
        if _MIN_CMP_RE.match(cmp_norm):
            lo = n + 1 if strict else n
        else:
            hi = max(0, n - 1) if strict else n
        t.consume(m)
    return lo, hi


_NEGATED_RE = re.compile(
    r"(?:\b(?:without|no|zero|free\s+of|excluding|exclude|except|not|non)\s*-?\s*(?:(?:any|the|a)\s+)?)$", re.I
)


def _parse_checks(t: _Text) -> str | None:
    """First check mentioned (by position). Negated mentions ("without PII") are consumed and ignored."""
    found: list[tuple[int, str]] = []
    for rx, name in _CHECK_RES:
        for m in list(rx.finditer(t.s)):
            neg = _NEGATED_RE.search(t.s[: m.start()])
            t.consume(m)
            if neg:
                t.s = t.s[: neg.start()] + " " * (neg.end() - neg.start()) + t.s[neg.end() :]
                continue
            found.append((m.start(), name))
    if not found:
        return None
    return min(found)[1]


def _parse_status(t: _Text) -> list[str] | None:
    statuses: set[str] = set()
    for rx, sts in _STATUS_RES:
        for m in list(rx.finditer(t.s)):
            statuses.update(sts)
            t.consume(m)
    return [s for s in STATUS_ORDER if s in statuses] or None


def _parse_category(t: _Text) -> str | None:
    for m in re.finditer(r"(?<![\w])(" + "|".join(CATEGORIES) + r")(?![\w])", t.s, re.I):
        t.consume(m)
        return m.group(1).lower()
    found: list[tuple[int, str]] = []
    for rx, cat in _CATEGORY_RES:
        for m in list(rx.finditer(t.s)):
            found.append((m.start(), cat))
            t.consume(m)
    return min(found)[1] if found else None


def _is_reserved(phrase: str) -> bool:
    """True if `phrase` is only status/check/category/length words (plus filler), not free text."""
    p = phrase.strip().lower()
    if not p:
        return True
    if re.fullmatch(r"[\d.,]+\s*k?\s*(?:tokens?|words?|chars?|characters?)?", p):
        return True
    probe = _Text(p)
    _parse_checks(probe)
    _parse_status(probe)
    _parse_category(probe)
    return all(w in _FILLER_WORDS for w in re.findall(r"[a-z0-9']+", probe.s))


def _parse_text_phrase(t: _Text, original: str) -> str | None:
    """'mentioning python', 'about tax law' -> the topic words (max 4), unless they are reserved."""
    src = original if len(original) == len(t.s) else t.s
    for m in _TEXT_RE.finditer(t.s):
        start = m.start("x")
        seg = t.s[start:]
        start += len(seg) - len(seg.lstrip())
        seg = t.s[start:]
        gap = re.search(r"\s{2,}", seg)  # a blanked-out (already parsed) region ends the phrase
        end = start + (gap.start() if gap else len(seg.rstrip()))
        raw = _TEXT_STOP.sub("", src[start:end])
        end = start + len(raw)
        raw = _LEADING_JUNK.sub("", raw).strip(" \t.,;:!?")
        phrase = " ".join(raw.split()[:4])
        if not phrase or _is_reserved(phrase):
            continue
        t.s = t.s[: m.start()] + " " * (end - m.start()) + t.s[end:]
        return phrase
    return None


def parse_rules(query: str, dataset_id: int | None = None) -> SampleFilter:
    """Deterministic keyword/regex parser. Always succeeds (possibly with an empty filter)."""
    q = " ".join(query.split())
    text_contains: str | None = None

    # 1. quoted phrases are literal text searches
    qm = _QUOTE_RE.search(q)
    if qm:
        text_contains = next(g for g in qm.groups() if g is not None).strip() or None
        q = q[: qm.start()] + " " * (qm.end() - qm.start()) + q[qm.end() :]
        q = _QUOTE_RE.sub(lambda m: " " * len(m.group(0)), q)

    original = q
    t = _Text(q.lower())

    # 2. explicit token / word / char bounds
    min_tokens, max_tokens = _parse_tokens(t)

    # 3. free-text topic ("mentioning python", "about the french revolution")
    if text_contains is None:
        text_contains = _parse_text_phrase(t, original)
    else:
        for trig in list(re.finditer(rf"\b(?:{_TEXT_TRIGGER})(?=\s{{2,}}|\s*$)", t.s, re.I)):
            t.consume(trig)

    # 4. quality checks (consumes "repeated characters", "too long", "failed the pii check", ...)
    failed_check = _parse_checks(t)

    # 5. "short answers" / "long samples"
    if max_tokens is None and (m := _SHORT_RE.search(t.s)):
        max_tokens = SHORT_ANSWER_MAX_TOKENS
        t.consume(m)
    if min_tokens is None and (m := _LONG_RE.search(t.s)):
        min_tokens = LONG_SAMPLE_MIN_TOKENS
        t.consume(m)

    # 6. qc status
    qc_status = _parse_status(t)

    # 7. category
    category = _parse_category(t)

    # 8. language
    lang: str | None = None
    if failed_check == "non_english":
        lang = "other"
    elif m := _ENGLISH_RE.search(t.s):
        lang = "en"
        t.consume(m)

    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        min_tokens, max_tokens = max_tokens, min_tokens

    return SampleFilter(
        dataset_id=dataset_id,
        qc_status=qc_status,
        category=category,
        lang=lang,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        text_contains=text_contains,
        failed_check=failed_check,
    )


# ---------------------------------------------------------------------------------------------
# LLM parser
# ---------------------------------------------------------------------------------------------

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

_FEW_SHOT = [
    ("failed samples with PII longer than 300 tokens",
     {"qc_status": ["fail"], "failed_check": "pii", "min_tokens": 301}),
    ("near duplicates in brainstorming", {"failed_check": "near_duplicate", "category": "brainstorming"}),
    ('refusals mentioning "OpenAI"', {"failed_check": "refusal_boilerplate", "text_contains": "OpenAI"}),
    ("clean summarization samples under 200 words",
     {"qc_status": ["pass"], "category": "summarization", "max_tokens": 259}),
]


def build_llm_prompt(query: str) -> str:
    schema = {
        "qc_status": "list of qc statuses or null; allowed: " + ", ".join(STATUS_ORDER),
        "category": "one canonical category or null; known: " + ", ".join(CATEGORIES),
        "lang": '"en" or "other" or null (non_english samples have lang "other")',
        "min_tokens": "integer or null (tokens ~= words * 1.3 ~= chars / 4)",
        "max_tokens": "integer or null",
        "text_contains": "literal substring to search in prompt/context/response, or null",
        "failed_check": "one check name or null; allowed: " + ", ".join(CHECK_NAMES),
    }
    lines = [
        "You convert a search query over an LLM fine-tuning dataset (instruction/response samples) into a JSON filter.",
        "Return ONLY a JSON object with these optional keys (omit or use null for unused ones):",
        json.dumps(schema, indent=2),
        "Rules: 'warnings/issues/flagged/needs review' -> qc_status [\"warn\",\"fail\"]; 'failed/broken/bad' -> [\"fail\"];"
        " 'clean/passed/good' -> [\"pass\"]; 'pending/unchecked' -> [\"pending\"]. Use only the allowed check names."
        " Leave category null for generic words like 'questions'. Never invent keys.",
        "Examples:",
    ]
    for q, out in _FEW_SHOT:
        lines.append(f"Query: {q}\nJSON: {json.dumps(out)}")
    lines.append(f"Query: {query}\nJSON:")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            return None
        text = text[a : b + 1]
    data = json.loads(text)
    return data if isinstance(data, dict) else None


def validate_llm_filter(data: dict, dataset_id: int | None) -> SampleFilter | None:
    """Strictly validate LLM output; None if anything looks off."""
    allowed = set(SampleFilter.model_fields) - {"dataset_id"}
    data = {k: v for k, v in data.items() if v is not None and k != "dataset_id"}
    if set(data) - allowed:
        return None
    if isinstance(data.get("qc_status"), str):
        data["qc_status"] = [data["qc_status"]]
    if data.get("failed_check") is not None and data["failed_check"] not in CHECK_NAMES:
        return None
    if isinstance(data.get("category"), str):
        c = re.sub(r"[\s-]+", "_", data["category"].strip().lower())
        data["category"] = c or None
    for k in ("min_tokens", "max_tokens"):
        v = data.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            return None
        if isinstance(v, float):
            data[k] = round(v)
    if isinstance(data.get("text_contains"), str):
        data["text_contains"] = data["text_contains"].strip() or None
    if data.get("lang") not in (None, "en", "other"):
        return None
    try:
        f = SampleFilter.model_validate({**data, "dataset_id": dataset_id})
    except ValidationError:
        return None
    if f.min_tokens is not None and f.max_tokens is not None and f.min_tokens > f.max_tokens:
        return None
    if f.qc_status:
        f.qc_status = [s for s in STATUS_ORDER if s in set(f.qc_status)]
    return f


def _call_gemini(key: str, prompt: str) -> str:
    r = httpx.post(
        GEMINI_URL,
        params={"key": key},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        },
        timeout=LLM_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(key: str, prompt: str) -> str:
    r = httpx.post(
        ANTHROPIC_URL,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 300,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=LLM_TIMEOUT_S,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


def parse_llm(query: str, dataset_id: int | None = None) -> SampleFilter | None:
    """Ask the configured LLM for a filter. Returns None when no key is set or on any error."""
    s = get_settings()
    providers = []
    if s.gemini_api_key:
        providers.append(("gemini", _call_gemini, s.gemini_api_key))
    if s.anthropic_api_key:
        providers.append(("anthropic", _call_anthropic, s.anthropic_api_key))
    if not providers:
        return None
    prompt = build_llm_prompt(query)
    for name, call, key in providers:
        try:
            data = _extract_json(call(key, prompt))
            if data is None:
                log.warning("nlsearch: %s returned non-object JSON", name)
                continue
            f = validate_llm_filter(data, dataset_id)
            if f is None:
                log.warning("nlsearch: %s returned an invalid filter: %s", name, data)
                continue
            return f
        except Exception as exc:  # network, HTTP status, bad JSON, unexpected shape ...
            log.warning("nlsearch: %s parser failed: %s", name, type(exc).__name__)
    return None


def _is_empty(f: SampleFilter) -> bool:
    return not any(v for k, v in f.model_dump().items() if k != "dataset_id")


def parse(query: str, dataset_id: int | None = None) -> tuple[SampleFilter, ParserName]:
    """LLM first (if configured), otherwise or on failure the rule-based parser.
    The dataset_id from the request always wins over anything the parser produced."""
    f = parse_llm(query, dataset_id)
    if f is not None:
        if _is_empty(f):
            rules = parse_rules(query, dataset_id)
            if not _is_empty(rules):
                return rules, "rules"
        f.dataset_id = dataset_id
        return f, "llm"
    f = parse_rules(query, dataset_id)
    f.dataset_id = dataset_id
    return f, "rules"

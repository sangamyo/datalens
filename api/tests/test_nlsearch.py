"""Natural-language search parser: rule-based phrasings, LLM path (mocked httpx) and fallback."""

import json

import httpx
import pytest

from app import nlsearch
from app.config import get_settings
from app.schemas import SampleFilter

# (query, expected non-None fields excluding dataset_id)
CASES = [
    # --- qc status ---
    ("failed samples", {"qc_status": ["fail"]}),
    ("broken samples", {"qc_status": ["fail"]}),
    ("bad ones", {"qc_status": ["fail"]}),
    ("samples with warnings", {"qc_status": ["warn", "fail"]}),
    ("anything with problems", {"qc_status": ["warn", "fail"]}),
    ("flagged samples", {"qc_status": ["warn", "fail"]}),
    ("warn or fail", {"qc_status": ["warn", "fail"]}),
    ("rows that need review", {"qc_status": ["warn", "fail"]}),
    ("only warnings", {"qc_status": ["warn"]}),
    ("clean samples", {"qc_status": ["pass"]}),
    ("passed", {"qc_status": ["pass"]}),
    ("High Quality examples", {"qc_status": ["pass"]}),
    ("samples with no issues", {"qc_status": ["pass"]}),
    ("not passed", {"qc_status": ["warn", "fail"]}),
    ("pending", {"qc_status": ["pending"]}),
    ("unchecked samples", {"qc_status": ["pending"]}),
    ("not checked yet", {"qc_status": ["pending"]}),
    ("failed or pending", {"qc_status": ["pending", "fail"]}),
    # --- checks ---
    ("duplicates", {"failed_check": "exact_duplicate"}),
    ("duplicated prompts", {"failed_check": "exact_duplicate"}),
    ("repeated samples", {"failed_check": "exact_duplicate"}),
    ("copies", {"failed_check": "exact_duplicate"}),
    ("near duplicates", {"failed_check": "near_duplicate"}),
    ("near-duplicate prompts", {"failed_check": "near_duplicate"}),
    ("similar samples", {"failed_check": "near_duplicate"}),
    ("almost the same prompts", {"failed_check": "near_duplicate"}),
    ("paraphrased duplicates", {"failed_check": "near_duplicate"}),
    ("PII", {"failed_check": "pii"}),
    ("personal data", {"failed_check": "pii"}),
    ("samples containing emails", {"failed_check": "pii"}),
    ("phone numbers", {"failed_check": "pii"}),
    ("sensitive info", {"failed_check": "pii"}),
    ("leaked secrets", {"failed_check": "pii"}),
    ("credit card numbers", {"failed_check": "pii"}),
    ("non-english samples", {"failed_check": "non_english", "lang": "other"}),
    ("Non English", {"failed_check": "non_english", "lang": "other"}),
    ("foreign language", {"failed_check": "non_english", "lang": "other"}),
    ("hindi", {"failed_check": "non_english", "lang": "other"}),
    ("other languages", {"failed_check": "non_english", "lang": "other"}),
    ("refusals", {"failed_check": "refusal_boilerplate"}),
    ("responses that say as an AI", {"failed_check": "refusal_boilerplate"}),
    ("boilerplate", {"failed_check": "refusal_boilerplate"}),
    ("canned responses", {"failed_check": "refusal_boilerplate"}),
    ("formatting issues", {"failed_check": "formatting"}),
    ("broken code", {"failed_check": "formatting"}),
    ("truncated responses", {"failed_check": "formatting"}),
    ("answers that got cut off", {"failed_check": "formatting"}),
    ("samples with repeated characters", {"failed_check": "formatting"}),
    ("empty responses", {"failed_check": "empty_or_short"}),
    ("blank answers", {"failed_check": "empty_or_short"}),
    ("too short", {"failed_check": "empty_or_short"}),
    ("one word answers", {"failed_check": "empty_or_short"}),
    ("too long", {"failed_check": "length_outlier"}),
    ("very long responses", {"failed_check": "length_outlier"}),
    ("outliers", {"failed_check": "length_outlier"}),
    ("unusual length", {"failed_check": "length_outlier"}),
    ("samples that failed the pii check", {"failed_check": "pii"}),
    ("clean samples without PII", {"qc_status": ["pass"]}),
    # --- tokens / length ---
    ("longer than 500 tokens", {"min_tokens": 501}),
    ("over 1k tokens", {"min_tokens": 1001}),
    ("at least 1.5k tokens", {"min_tokens": 1500}),
    ("under 50 tokens", {"max_tokens": 49}),
    ("at most 80 tokens", {"max_tokens": 80}),
    ("between 100 and 300 tokens", {"min_tokens": 100, "max_tokens": 300}),
    ("100-300 tokens", {"min_tokens": 100, "max_tokens": 300}),
    ("500+ tokens", {"min_tokens": 500}),
    ("about 200 tokens", {"min_tokens": 160, "max_tokens": 240}),
    ("longer than 500", {"min_tokens": 501}),
    ("longer than 200 words", {"min_tokens": 261}),
    ("under 2000 characters", {"max_tokens": 499}),
    ("between 400 and 800 chars", {"min_tokens": 100, "max_tokens": 200}),
    ("short answers", {"max_tokens": 40}),
    ("long answers", {"min_tokens": 400}),
    ("long samples", {"min_tokens": 400}),
    ("long creative writing samples", {"min_tokens": 400, "category": "creative_writing"}),
    ("short classification answers", {"max_tokens": 40, "category": "classification"}),
    # --- category ---
    ("brainstorm", {"category": "brainstorming"}),
    ("brainstorming samples", {"category": "brainstorming"}),
    ("creative writing", {"category": "creative_writing"}),
    ("stories", {"category": "creative_writing"}),
    ("poems", {"category": "creative_writing"}),
    ("summaries", {"category": "summarization"}),
    ("summarization", {"category": "summarization"}),
    ("open qa", {"category": "open_qa"}),
    ("closed_qa", {"category": "closed_qa"}),
    ("general qa", {"category": "general_qa"}),
    ("classification", {"category": "classification"}),
    ("information extraction", {"category": "information_extraction"}),
    ("rewrite tasks", {"category": "rewrite"}),
    ("chat", {"category": "chat"}),
    ("coding", {"category": "coding"}),
    ("generation", {"category": "generation"}),
    ("questions", {}),
    ("QA", {}),
    # --- text ---
    ('"photosynthesis"', {"text_contains": "photosynthesis"}),
    ("mentioning taxes", {"text_contains": "taxes"}),
    ("containing recipe", {"text_contains": "recipe"}),
    ("samples about python", {"text_contains": "python"}),
    ("that talk about the french revolution", {"text_contains": "french revolution"}),
    ("samples about duplicates", {"failed_check": "exact_duplicate"}),
    ("show me everything", {}),
    # --- combinations ---
    ("failed samples with PII longer than 300 tokens", {"qc_status": ["fail"], "failed_check": "pii", "min_tokens": 301}),
    ("near duplicates in brainstorming", {"failed_check": "near_duplicate", "category": "brainstorming"}),
    ('refusals mentioning "OpenAI"', {"failed_check": "refusal_boilerplate", "text_contains": "OpenAI"}),
    ("clean summarization samples under 200 words", {"qc_status": ["pass"], "category": "summarization", "max_tokens": 259}),
    ("summaries over 300 tokens with issues",
     {"qc_status": ["warn", "fail"], "category": "summarization", "min_tokens": 301}),
    ("short answers in classification with PII", {"category": "classification", "max_tokens": 40, "failed_check": "pii"}),
    ("long samples about climate change", {"min_tokens": 400, "text_contains": "climate change"}),
    ("poems about cats", {"category": "creative_writing", "text_contains": "cats"}),
    ("containing pizza in creative writing", {"category": "creative_writing", "text_contains": "pizza"}),
    ("coding samples with broken code", {"category": "coding", "failed_check": "formatting"}),
    ("classification samples that need review", {"qc_status": ["warn", "fail"], "category": "classification"}),
    ("refusals that start with 'I am sorry'", {"failed_check": "refusal_boilerplate", "text_contains": "I am sorry"}),
    ("passed QC but longer than 2k tokens", {"qc_status": ["pass"], "min_tokens": 2001}),
    ("english only", {"lang": "en"}),
    ("FAILED NON-ENGLISH OPEN QA", {"qc_status": ["fail"], "failed_check": "non_english", "lang": "other", "category": "open_qa"}),
]


@pytest.mark.parametrize("query,expected", CASES, ids=[c[0] for c in CASES])
def test_parse_rules(query, expected):
    f = nlsearch.parse_rules(query, 7)
    got = f.model_dump(exclude_none=True)
    assert got.pop("dataset_id") == 7
    assert got == expected


def test_parse_rules_output_is_valid_filter():
    for query, _ in CASES:
        f = nlsearch.parse_rules(query)
        SampleFilter.model_validate(f.model_dump())
        assert f.failed_check is None or f.failed_check in nlsearch.CHECK_NAMES
        assert f.category is None or f.category in nlsearch.CATEGORIES


def test_parse_rules_has_at_least_40_phrasings():
    assert len(CASES) >= 40


@pytest.mark.parametrize(
    "canonical,available,expected",
    [
        ("brainstorming", ["Brainstorm", "Chat"], "Brainstorm"),
        ("open_qa", ["Open QA", "Closed QA"], "Open QA"),
        ("summarization", ["Summarize"], "Summarize"),
        ("information_extraction", ["Extract"], "Extract"),
        ("open_qa", ["open_qa", "closed_qa"], "open_qa"),
        ("coding", ["brainstorming"], "coding"),
    ],
)
def test_resolve_category(canonical, available, expected):
    assert nlsearch.resolve_category(canonical, available) == expected


# ---------------------------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def no_keys(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", None)
    monkeypatch.setattr(s, "anthropic_api_key", None)
    return s


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=httpx.Request("POST", "http://x"), response=None)

    def json(self):
        return self._payload


def _gemini_payload(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _anthropic_payload(text):
    return {"content": [{"type": "text", "text": text}]}


def test_parse_without_keys_uses_rules(no_keys, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("no HTTP call expected")

    monkeypatch.setattr(httpx, "post", fail)
    f, parser = nlsearch.parse("failed samples with PII", 3)
    assert parser == "rules"
    assert f.dataset_id == 3 and f.failed_check == "pii" and f.qc_status == ["fail"]
    assert nlsearch.parse_llm("anything", 3) is None


def test_gemini_path(no_keys, monkeypatch):
    no_keys.gemini_api_key = "g-key"
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        out = {"qc_status": ["fail"], "failed_check": "pii", "min_tokens": 301, "dataset_id": 99}
        return _Resp(_gemini_payload(json.dumps(out)))

    monkeypatch.setattr(httpx, "post", post)
    f, parser = nlsearch.parse("failed samples with PII longer than 300 tokens", 5)
    assert parser == "llm"
    assert f.dataset_id == 5  # request dataset_id wins
    assert f.failed_check == "pii" and f.min_tokens == 301 and f.qc_status == ["fail"]
    url, kw = calls[0]
    assert "gemini-2.0-flash:generateContent" in url
    assert kw["params"] == {"key": "g-key"}
    assert kw["timeout"] == 10
    assert kw["json"]["generationConfig"]["responseMimeType"] == "application/json"
    prompt = kw["json"]["contents"][0]["parts"][0]["text"]
    assert "failed samples with PII longer than 300 tokens" in prompt
    assert "refusal_boilerplate" in prompt and "brainstorming" in prompt


def test_anthropic_path(no_keys, monkeypatch):
    no_keys.anthropic_api_key = "a-key"
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        return _Resp(_anthropic_payload('```json\n{"failed_check": "near_duplicate", "category": "Brainstorming"}\n```'))

    monkeypatch.setattr(httpx, "post", post)
    f, parser = nlsearch.parse("near duplicates in brainstorming", None)
    assert parser == "llm"
    assert f.failed_check == "near_duplicate" and f.category == "brainstorming" and f.dataset_id is None
    url, kw = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert kw["headers"]["x-api-key"] == "a-key"
    assert kw["headers"]["anthropic-version"] == "2023-06-01"
    assert kw["json"]["model"] == "claude-haiku-4-5-20251001"
    assert kw["json"]["max_tokens"] == 300
    assert kw["timeout"] == 10


def test_gemini_failure_falls_through_to_anthropic(no_keys, monkeypatch):
    no_keys.gemini_api_key = "g"
    no_keys.anthropic_api_key = "a"

    def post(url, **kw):
        if "googleapis" in url:
            return _Resp({}, status=500)
        return _Resp(_anthropic_payload('{"qc_status": "pass"}'))

    monkeypatch.setattr(httpx, "post", post)
    f, parser = nlsearch.parse("clean", 1)
    assert parser == "llm" and f.qc_status == ["pass"]


@pytest.mark.parametrize(
    "text",
    [
        '{"failed_check": "toxicity"}',  # unknown check
        '{"qc_status": ["great"]}',  # unknown status
        '{"min_tokens": -5}',
        '{"min_tokens": "lots"}',
        '{"min_tokens": 500, "max_tokens": 10}',
        '{"lang": "fr"}',
        '{"sql": "DROP TABLE samples"}',  # unknown key
        "not json at all",
        "[1, 2, 3]",
    ],
)
def test_invalid_llm_output_returns_none_and_falls_back(no_keys, monkeypatch, text):
    no_keys.gemini_api_key = "g"
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp(_gemini_payload(text)))
    assert nlsearch.parse_llm("failed samples", 1) is None
    f, parser = nlsearch.parse("failed samples", 1)
    assert parser == "rules" and f.qc_status == ["fail"]


def test_llm_network_error_falls_back(no_keys, monkeypatch):
    no_keys.anthropic_api_key = "a"

    def post(url, **kw):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(httpx, "post", post)
    f, parser = nlsearch.parse("duplicates", 2)
    assert parser == "rules" and f.failed_check == "exact_duplicate" and f.dataset_id == 2


def test_llm_unexpected_shape_returns_none(no_keys, monkeypatch):
    no_keys.gemini_api_key = "g"
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp({"candidates": []}))
    assert nlsearch.parse_llm("x", None) is None


def test_empty_llm_filter_prefers_nonempty_rules(no_keys, monkeypatch):
    no_keys.gemini_api_key = "g"
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp(_gemini_payload("{}")))
    f, parser = nlsearch.parse("duplicates", 1)
    assert parser == "rules" and f.failed_check == "exact_duplicate"


def test_prompt_contains_only_the_query():
    p = nlsearch.build_llm_prompt("my query")
    assert p.rstrip().endswith("Query: my query\nJSON:")
    assert p.count("Query:") == 5  # 4 few-shot examples + the query

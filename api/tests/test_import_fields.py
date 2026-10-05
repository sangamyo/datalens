"""Field auto-detection, chat parsing and text helpers of the import job (no network, no DB)."""

import hashlib
import json

import pyarrow as pa
import pytest

from app.jobs.import_job import (
    DatasetImportError,
    build_sample,
    class_label_names,
    content_hash,
    convert_configs,
    convert_splits,
    detect_fields,
    estimate_tokens,
    extract_row,
    norm_text,
    parse_chat,
    pick_default_config,
    select_convert_files,
    select_main_files,
)

S = pa.string()
MSG = pa.list_(pa.struct([("content", S), ("role", S)]))
SHAREGPT = pa.list_(pa.struct([("from", S), ("value", S)]))


def schema(**cols: pa.DataType) -> pa.Schema:
    return pa.schema(list(cols.items()))


# ---------------------------------------------------------------- the four reference datasets
def test_dolly_shape() -> None:
    sc = schema(instruction=S, context=S, response=S, category=S)
    assert detect_fields(sc) == {"prompt": "instruction", "context": "context", "response": "response", "category": "category"}


def test_alpaca_shape_input_is_context() -> None:
    sc = schema(instruction=S, input=S, output=S, text=S)
    assert detect_fields(sc) == {"prompt": "instruction", "context": "input", "response": "output"}


def test_no_robots_shape_is_chat() -> None:
    # `prompt` exists but there is no response column -> chat format from `messages`.
    sc = schema(prompt=S, prompt_id=S, messages=MSG, category=S)
    assert detect_fields(sc) == {"messages": "messages", "category": "category"}


def test_gsm8k_shape() -> None:
    assert detect_fields(schema(question=S, answer=S)) == {"prompt": "question", "response": "answer"}


# ---------------------------------------------------------------- other shapes
def test_input_as_prompt_is_not_also_context() -> None:
    assert detect_fields(schema(input=S, output=S)) == {"prompt": "input", "response": "output"}


def test_prompt_completion_and_case_insensitive() -> None:
    assert detect_fields(schema(Prompt=S, Completion=S, Source=S)) == {
        "prompt": "Prompt",
        "response": "Completion",
        "category": "Source",
    }


def test_chosen_as_response_and_int_label_category() -> None:
    assert detect_fields(schema(prompt=S, chosen=S, rejected=S, label=pa.int64())) == {
        "prompt": "prompt",
        "response": "chosen",
        "category": "label",
    }


def test_sharegpt_conversations() -> None:
    assert detect_fields(schema(id=S, conversations=SHAREGPT)) == {"messages": "conversations"}


def test_explicit_mapping_wins() -> None:
    sc = schema(q=S, a=S, instruction=S, response=S, topic=S)
    assert detect_fields(sc, {"prompt": "q", "response": "a", "category": "topic", "context": None}) == {
        "prompt": "q",
        "response": "a",
        "category": "topic",
    }


def test_explicit_mapping_unknown_column() -> None:
    with pytest.raises(DatasetImportError, match="column\\(s\\) not found: nope; available columns: q, a"):
        detect_fields(schema(q=S, a=S), {"prompt": "nope", "response": "a"})


def test_explicit_chat_mapping() -> None:
    assert detect_fields(schema(dialog=MSG), {"messages": "dialog"}) == {"messages": "dialog"}


def test_failure_lists_available_columns() -> None:
    with pytest.raises(DatasetImportError) as exc:
        detect_fields(schema(text=S, id=pa.int64()))
    assert str(exc.value) == "no prompt/response columns found; available columns: text, id"


def test_non_text_prompt_column_is_ignored() -> None:
    with pytest.raises(DatasetImportError):
        detect_fields(schema(prompt=pa.int64(), response=S))


def test_class_label_names_from_hf_metadata() -> None:
    meta = {"info": {"features": {"label": {"names": ["neg", "pos"], "_type": "ClassLabel"}, "text": {"_type": "Value"}}}}
    sc = schema(text=S, label=pa.int64()).with_metadata({b"huggingface": json.dumps(meta).encode()})
    labels = class_label_names(sc)
    assert labels == {"label": ["neg", "pos"]}
    fields = {"prompt": "text", "response": "text", "category": "label"}
    assert extract_row({"text": "hi", "label": 1}, fields, labels)[3] == "pos"


# ---------------------------------------------------------------- chat parsing
def test_parse_chat_role_content_with_system() -> None:
    msgs = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": " Hi "},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "second"},
    ]
    assert parse_chat(msgs) == ("Hi", "Hello!", "Be terse.")


def test_parse_chat_sharegpt() -> None:
    msgs = [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]
    assert parse_chat(msgs) == ("Q?", "A.", None)


def test_parse_chat_assistant_before_user_is_skipped_and_missing_reply() -> None:
    assert parse_chat([{"role": "assistant", "content": "greeting"}, {"role": "user", "content": "q"}]) == ("q", "", None)
    assert parse_chat(None) == ("", "", None)


def test_parse_chat_json_string_and_content_parts() -> None:
    msgs = json.dumps([{"role": "user", "content": [{"type": "text", "text": "look"}]}, {"role": "assistant", "content": "ok"}])
    assert parse_chat(msgs) == ("look", "ok", None)


def test_extract_row_chat_and_instruction() -> None:
    row = {"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}], "category": "Chat"}
    assert extract_row(row, {"messages": "messages", "category": "category"}) == ("p", None, "r", "Chat")
    row = {"instruction": "i", "input": "", "output": "o"}
    assert extract_row(row, {"prompt": "instruction", "context": "input", "response": "output"}) == ("i", "", "o", None)


# ---------------------------------------------------------------- text helpers
def test_norm_text() -> None:
    assert norm_text(None) == ""
    assert norm_text("  a b \n") == "a b"
    assert norm_text(42) == "42"
    assert norm_text("x\x00y") == "xy"
    assert norm_text(b"bytes") == "bytes"


def test_content_hash_normalises_whitespace_and_case() -> None:
    a = content_hash("Hello   World", None, "Fine\n\nthanks")
    assert a == content_hash("hello world", "", "fine thanks")
    assert a == hashlib.sha1("hello world\n\nfine thanks".encode()).hexdigest()
    assert a != content_hash("hello world", "ctx", "fine thanks")


def test_estimate_tokens() -> None:
    assert estimate_tokens("", None, "") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcd", "e", None) == 2


def test_build_sample() -> None:
    s = build_sample(7, 3, " What is 2+2? ", "", "4", None)
    assert s["dataset_id"] == 7 and s["sample_index"] == 3
    assert s["prompt"] == "What is 2+2?" and s["context"] is None and s["category"] is None
    assert (s["prompt_chars"], s["response_chars"], s["tokens_est"]) == (12, 1, 4)
    assert s["qc_status"] == "pending" and len(s["content_hash"]) == 40
    assert build_sample(1, 0, "p", "c", "r", "x" * 300)["category"] == "x" * 100


# ---------------------------------------------------------------- file selection
CONVERT_GSM8K = [
    ".gitattributes",
    "main/test/0000.parquet",
    "main/train/0000.parquet",
    "socratic/test/0000.parquet",
    "socratic/train/0000.parquet",
]


def test_convert_listing_helpers() -> None:
    assert convert_configs(CONVERT_GSM8K) == ["main", "socratic"]
    assert convert_splits(CONVERT_GSM8K, "main") == ["test", "train"]
    assert select_convert_files(CONVERT_GSM8K, "main", "train") == ["main/train/0000.parquet"]
    assert select_convert_files(CONVERT_GSM8K, "main", "validation") == []


def test_convert_partial_and_flat_layouts() -> None:
    files = ["default/partial-train/0000.parquet", "default/partial-train/0001.parquet"]
    assert select_convert_files(files, "default", "train") == files
    flat = ["cfg/train-00000-of-00002.parquet", "cfg/train-00001-of-00002.parquet", "cfg/test.parquet"]
    assert select_convert_files(flat, "cfg", "train") == flat[:2]
    assert convert_splits(flat, "cfg") == ["train", "test"]


def test_pick_default_config() -> None:
    assert pick_default_config(["default"]) == "default"
    assert pick_default_config(["main", "socratic"], [("main", False), ("socratic", False)]) == "main"
    assert pick_default_config(["a", "b"], [("a", False), ("b", True)]) == "b"
    assert pick_default_config([], []) == "default"


def test_select_main_files() -> None:
    no_robots = [
        "README.md",
        "data/test-00000-of-00001.parquet",
        "data/train-00000-of-00001.parquet",
        "data/train_sft-00000-of-00001.parquet",
    ]
    assert select_main_files(no_robots, "default", "train") == ["data/train-00000-of-00001.parquet"]
    assert select_main_files(no_robots, "default", "train_sft") == ["data/train_sft-00000-of-00001.parquet"]
    assert select_main_files(["README.md", "databricks-dolly-15k.jsonl"], "default", "train") == ["databricks-dolly-15k.jsonl"]
    assert select_main_files(["README.md", "databricks-dolly-15k.jsonl"], "default", "test") == []
    gsm8k = ["main/train-00000-of-00001.parquet", "socratic/train-00000-of-00001.parquet"]
    assert select_main_files(gsm8k, "main", "train") == ["main/train-00000-of-00001.parquet"]
    # parquet preferred over json for the same split
    assert select_main_files(["train.json", "train.parquet"], "default", "train") == ["train.parquet"]

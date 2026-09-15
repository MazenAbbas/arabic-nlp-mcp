import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from arabic_nlp_mcp.search import prepare_batch, prepare_for_search


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("إِنَّ الكتاب", "ان الكتاب"),
        ("علي\u200fكم", "عليكم"),
        ("رقم ١٢٣", "رقم 123"),
        ("نِظَام", "نظام"),
        ("ســلام", "سلام"),
        ("AI بالعربي", "ai بالعربي"),
        ("ﻻ بأس", "لا باس"),
    ],
)
def test_search_profile_equivalent_forms_have_same_fingerprint(left, right):
    assert prepare_for_search(left).fingerprint == prepare_for_search(right).fingerprint


@pytest.mark.parametrize(
    ("left", "right"),
    [("كتاب", "کتاب"), ("عليكم", "علیکم"), ("نص\x00آمن", "نصآمن")],
)
def test_persian_variants_and_controls_normalize_for_search(left, right):
    assert prepare_for_search(left).fingerprint == prepare_for_search(right).fingerprint


@pytest.mark.parametrize("profile", ["conservative", "search", "aggressive"])
def test_search_preparation_is_idempotent(profile):
    once = prepare_for_search("  إِنَّ هــذا جمييييل  ", profile=profile)
    twice = prepare_for_search(once.search_text, profile=profile)
    assert twice.search_text == once.search_text
    assert twice.fingerprint == once.fingerprint


def test_conservative_preserves_letters_marks_and_digits():
    result = prepare_for_search("إِنَّ ١٢٣", profile="conservative")
    assert result.search_text == "إِنَّ ١٢٣"


def test_aggressive_is_explicitly_lossy():
    result = prepare_for_search("مدرسة جمييييلة", profile="aggressive")
    assert result.search_text == "مدرسه جمييله"
    assert result.warnings
    assert result.lossy is True


def test_conservative_is_marked_non_lossy():
    assert prepare_for_search("نص", profile="conservative").lossy is False


def test_search_tokens_preserve_mixed_language_and_numbers():
    result = prepare_for_search("نظام GPT-5 رقم ٢٠٢٦")
    assert result.tokens == ["نظام", "gpt", "5", "رقم", "2026"]


def test_empty_input_has_stable_empty_fingerprint():
    result = prepare_for_search("")
    assert result.tokens == []
    assert len(result.fingerprint) == 64


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match="profile"):
        prepare_for_search("نص", profile="unsafe")


def test_cli_emits_utf8_json():
    completed = subprocess.run(
        [sys.executable, "-m", "arabic_nlp_mcp.cli", "search", "إِنَّ AI"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(completed.stdout)["search_text"] == "ان ai"


def test_cli_reads_stdin():
    completed = subprocess.run(
        [sys.executable, "-m", "arabic_nlp_mcp.cli", "search"],
        input="مَرْحَبًا",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(completed.stdout)["search_text"] == "مرحبا"


def test_batch_is_lazy_and_ordered():
    results = prepare_batch(text for text in ["أول", "ثانٍ"])
    assert [item.search_text for item in results] == ["اول", "ثان"]


def test_cli_jsonl_streaming():
    completed = subprocess.run(
        [sys.executable, "-m", "arabic_nlp_mcp.cli", "search", "--jsonl"],
        input="إِنَّ\nرقم ١٢\n",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [row["search_text"] for row in rows] == ["ان", "رقم 12"]


def test_search_is_safe_under_concurrent_calls():
    texts = [f"إِنَّ الطلب رقم {index} آمن" for index in range(1_000)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(prepare_for_search, texts))
    assert len({result.fingerprint for result in results}) == len(texts)

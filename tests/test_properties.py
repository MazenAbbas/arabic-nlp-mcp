import re

from hypothesis import given, settings
from hypothesis import strategies as st

from arabic_nlp_mcp.common import CONTROL_CHAR_RE, FORMAT_CONTROL_RE
from arabic_nlp_mcp.diacritize import diacritize
from arabic_nlp_mcp.dialect import detect_dialect
from arabic_nlp_mcp.normalize import normalize
from arabic_nlp_mcp.search import prepare_for_search
from arabic_nlp_mcp.sentiment import analyze_sentiment

UNICODE_TEXT = st.text(max_size=300)
ARABIC_TEXT = st.text(
    alphabet=st.characters(min_codepoint=0x600, max_codepoint=0x6FF), max_size=300
)


@settings(max_examples=300, deadline=None)
@given(UNICODE_TEXT, st.sampled_from(("conservative", "search", "aggressive")))
def test_search_is_deterministic_and_idempotent(text, profile):
    first = prepare_for_search(text, profile=profile)
    repeated = prepare_for_search(text, profile=profile)
    normalized_again = prepare_for_search(first.search_text, profile=profile)
    assert first == repeated
    assert normalized_again.search_text == first.search_text
    assert normalized_again.fingerprint == first.fingerprint


@settings(max_examples=300, deadline=None)
@given(UNICODE_TEXT)
def test_stable_tools_remove_unsafe_controls(text):
    search = prepare_for_search(text)
    normalized = normalize(text)
    assert CONTROL_CHAR_RE.search(search.search_text) is None
    assert FORMAT_CONTROL_RE.search(search.search_text) is None
    assert CONTROL_CHAR_RE.search(normalized.normalized) is None
    assert FORMAT_CONTROL_RE.search(normalized.normalized) is None


@settings(max_examples=250, deadline=None)
@given(ARABIC_TEXT)
def test_all_local_backends_accept_bounded_arabic_unicode(text):
    # This is a crash/property corpus, not an accuracy benchmark.
    normalize(text)
    prepare_for_search(text)
    detect_dialect(text)
    analyze_sentiment(text, backend="lexicon")
    diacritize(text)


@settings(max_examples=200, deadline=None)
@given(UNICODE_TEXT)
def test_fingerprint_is_lowercase_sha256(text):
    fingerprint = prepare_for_search(text).fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)

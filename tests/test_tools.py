import pytest
from pydantic import ValidationError

from arabic_nlp_mcp.common import MAX_TEXT_LENGTH
from arabic_nlp_mcp.diacritize import diacritize
from arabic_nlp_mcp.dialect import detect_dialect
from arabic_nlp_mcp.models import DiacritizeResult, DialectResult, NormalizeResult, SentimentResult
from arabic_nlp_mcp.normalize import normalize
from arabic_nlp_mcp.sentiment import analyze_sentiment


@pytest.mark.parametrize("value", [None, 1, [], {}])
@pytest.mark.parametrize("function", [normalize, detect_dialect, analyze_sentiment, diacritize])
def test_rejects_non_string_input(function, value):
    with pytest.raises(TypeError):
        function(value)


@pytest.mark.parametrize("function", [normalize, detect_dialect, analyze_sentiment, diacritize])
def test_rejects_oversized_input(function):
    with pytest.raises(ValueError, match="at most"):
        function("ا" * (MAX_TEXT_LENGTH + 1))


def test_normalize_contract_and_rules():
    result = normalize("  إنَّـــ رائع!!!  ")
    assert isinstance(result, NormalizeResult)
    assert result.normalized == "ان رائع!!!"
    assert result.changed
    assert result.lossy and result.warnings


def test_normalize_only_collapses_arabic_letters():
    assert normalize("جمييييل!!!!").normalized == "جمييل!!!!"


def test_normalize_options_can_be_disabled():
    assert normalize("إِنَّ", strip_diacritics=False, unify_alef=False).normalized == "إِنَّ"


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("وش تسوي الحين", "gulf"),
        ("عايز اروح فين", "egyptian"),
        ("شو بدك تعمل هلأ", "levantine"),
        ("وعليه فإن النتيجة مهمة", "msa"),
    ],
)
def test_dialect_distinctive_markers(text, label):
    assert detect_dialect(text).predicted == label


def test_dialect_no_evidence_is_unknown():
    result = detect_dialect("الطاولة والكرسي والباب")
    assert isinstance(result, DialectResult)
    assert result.predicted == "unknown" and result.evidence_score == 0


def test_dialect_shared_marker_is_mixed():
    assert detect_dialect("وين").predicted == "mixed"


def test_dialect_never_matches_substrings():
    assert "مش" not in detect_dialect("هذا مشروع جديد").matched_terms["egyptian"]


def test_dialect_phrase_requires_adjacent_tokens():
    assert "يا هلا" in detect_dialect("يا هلا وسهلا").matched_terms["gulf"]
    assert "يا هلا" not in detect_dialect("يا صاحبي هلا").matched_terms["gulf"]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("هذا المنتج رائع وممتاز", "positive"),
        ("مش حلو خالص", "negative"),
        ("تجربة عادية", "neutral"),
        ("ما شاء الله عليك", "positive"),
    ],
)
def test_lexicon_sentiment(text, label):
    result = analyze_sentiment(text, backend="lexicon")
    assert isinstance(result, SentimentResult)
    assert result.label == label


def test_sentiment_bad_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        analyze_sentiment("رائع", backend="magic")


def test_explicit_transformer_does_not_silently_fallback(monkeypatch):
    import arabic_nlp_mcp.sentiment as module

    monkeypatch.setattr(module, "_get_pipeline", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        module.analyze_sentiment("رائع", backend="transformer")


def test_auto_reports_fallback(monkeypatch):
    import arabic_nlp_mcp.sentiment as module

    monkeypatch.setattr(module, "_get_pipeline", lambda: None)
    assert module.analyze_sentiment("رائع", backend="auto").fallback_reason


def test_transformer_uses_tokenizer_truncation(monkeypatch):
    import arabic_nlp_mcp.sentiment as module

    calls = []

    def fake(text, **kwargs):
        calls.append((text, kwargs))
        return [{"label": "POSITIVE", "score": 0.9}]

    monkeypatch.setattr(module, "_get_pipeline", lambda: fake)
    assert module.analyze_sentiment("رائع", backend="transformer").label == "positive"
    assert calls == [("رائع", {"truncation": True})]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("هذا كتاب جميل", "هَذَا كِتَاب جَمِيل"),
        ("الكتاب", "الْكِتَاب"),
        ("الشمس", "ال" + "ش\u0651\u064eمْس"),
        ("بطيئة قوية", "بَطِيئَة قَوِيَّة"),
        ("كِتاب", "كِتَاب"),
    ],
)
def test_lexical_diacritization(text, expected):
    result = diacritize(text)
    assert isinstance(result, DiacritizeResult)
    assert result.diacritized == expected


def test_diacritize_unknown_is_unchanged():
    result = diacritize("زخرفشتوكولوجيا")
    assert result.diacritized == "زخرفشتوكولوجيا" and result.coverage == 0


def test_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        NormalizeResult(
            original="",
            normalized="",
            changed=False,
            lossy=False,
            transformations_applied=[],
            surprise=True,
        )

"""Arabic sentiment analysis.

Hybrid design: tries a real pretrained transformer model first
(CAMeL-Lab's Arabic BERT fine-tuned for sentiment), and falls back to a
lexicon-based polarity scorer if the model can't be loaded - no internet
access, no local cache, out-of-memory, whatever. This was built in a
sandboxed environment with no access to huggingface.co, so the model path
could not be exercised end-to-end here; it *will* run normally on a
machine with regular internet access, which is why it stays as the first
attempt rather than being ripped out. Either way, the result always states
which method actually produced it - never silently pretends the lexicon
fallback is model output, or vice versa.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Protocol, cast

from .common import validate_text
from .models import SentimentLabel, SentimentResult

_MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"


class _Pipeline(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, Any]]: ...


_pipeline: _Pipeline | None = None
_pipeline_lock = threading.Lock()
_pipeline_load_failed = False
_pipeline_failure_reason: str | None = None


def _pipeline_state() -> tuple[_Pipeline | None, bool]:
    return _pipeline, _pipeline_load_failed


def _get_pipeline() -> _Pipeline | None:
    """Lazily load the HF sentiment pipeline once, thread-safely.

    Returns None (and remembers not to retry) if it can't be loaded, so
    every call after the first failure skips straight to the lexicon path
    instead of retrying a slow network call every time.
    """
    global _pipeline, _pipeline_load_failed, _pipeline_failure_reason
    current, failed = _pipeline_state()
    if current is not None or failed:
        return current

    with _pipeline_lock:
        current, failed = _pipeline_state()
        if current is not None or failed:
            return current
        try:
            from transformers import pipeline  # local import: heavy, optional

            _pipeline = cast(
                _Pipeline, pipeline("sentiment-analysis", model=_MODEL_ID, tokenizer=_MODEL_ID)
            )
        except (ImportError, OSError, RuntimeError) as exc:
            _pipeline_load_failed = True
            _pipeline = None
            _pipeline_failure_reason = f"{type(exc).__name__}: {exc}"
    return _pipeline


# --- Lexicon fallback -------------------------------------------------

_POSITIVE = {
    # Core quality/praise
    "جميل",
    "جميلة",
    "رائع",
    "رائعة",
    "ممتاز",
    "ممتازة",
    "عظيم",
    "عظيمة",
    "رهيب",
    "رهيبة",
    "خرافي",
    "خرافية",
    "احترافي",
    "احترافية",
    "مبدع",
    "مبدعة",
    "إبداع",
    "ابداع",
    "متقن",
    "متقنة",
    "فخم",
    "فخمة",
    "انيق",
    "أنيق",
    "انيقة",
    "أنيقة",
    "مريح",
    "مريحة",
    "سلس",
    "سلسة",
    "دقيق",
    "دقيقة",
    "نظيف",
    "نظيفة",
    "سريع",
    "سريعة",
    "قوي",
    "قوية",
    # Emotion / reaction
    "احب",
    "أحب",
    "بحب",
    "احبه",
    "أحبه",
    "احبها",
    "أحبها",
    "سعيد",
    "سعيدة",
    "فرحان",
    "فرحانة",
    "فرح",
    "فرحة",
    "مبسوط",
    "مبسوطة",
    "مرتاح",
    "مرتاحة",
    "متحمس",
    "متحمسة",
    "معجب",
    "معجبة",
    "فخور",
    "فخورة",
    "مبهور",
    "مبهورة",
    "منبهر",
    "منبهرة",
    "استمتعت",
    "استمتعنا",
    "انبهرت",
    "أنبهرت",
    # Gulf/Saudi register
    "زين",
    "زينة",
    "حلو",
    "حلوة",
    "تمام",
    "طيب",
    "طيبة",
    "وايد حلو",
    "ما شاء الله",
    "ماشاءالله",
    "يعطيك العافية",
    "يعطيكم العافية",
    "تسلم",
    "تسلمين",
    "الله يعطيك العافية",
    "خوش",
    "مره حلو",
    "مرة حلو",
    # Egyptian/Levantine register
    "منيح",
    "منيحة",
    "كويس",
    "كويسة",
    "جامد",
    "جامدة",
    "حلوين",
    "قشطة",
    "تحفة",
    "عسل",
    "فل",
    "تمام كده",
    # Success / outcome
    "ناجح",
    "ناجحة",
    "نجاح",
    "نجحنا",
    "انجاز",
    "إنجاز",
    "وصلنا",
    "تفوق",
    "تفوقنا",
    "الحمدلله",
    "الحمد لله",
    "الحمدالله",
    "مبروك",
    "الف مبروك",
    "ألف مبروك",
    "شكرا",
    "شكرًا",
    "شكرا جزيلا",
    "شكراً جزيلاً",
    "احسنت",
    "أحسنت",
    "برافو",
    "يسلمو",
    "يسلموا",
    # Recommendation / trust signals (common in reviews)
    "انصح",
    "أنصح",
    "انصح به",
    "انصحكم",
    "يستاهل",
    "يستحق",
    "يستاهل التجربة",
    "يستاهل السعر",
    "يفوق التوقعات",
    "فاق التوقعات",
    "الافضل",
    "الأفضل",
    "افضل",
    "أفضل",
    "احلى",
    "أحلى",
    "لا يفوتكم",
    "تجربة رائعة",
    "تجربة ممتازة",
    "خدمة ممتازة",
    "خدمة رائعة",
    # Work / business
    "منتج",
    "منتجة",
    "فعال",
    "فعالة",
    "كفوء",
    "كفوءة",
    "ملتزم",
    "ملتزمة",
    "منظم",
    "منظمة",
    "مبتكر",
    "مبتكرة",
    "طموح",
    "طموحة",
    "موثوق",
    "موثوقة",
    "محترف",
    "محترفة",
    "متعاون",
    "متعاونة",
    "مسؤول",
    "مسؤولة",
    "ربح",
    "ارباح",
    "أرباح",
    "نمو",
    "توسع",
    "انطلاقة",
    "إنطلاقة",
    "ترقية",
    "علاوة",
    "تقدير",
    "شهادة تقدير",
    # Sports / competition
    "فوز",
    "فاز",
    "فازوا",
    "بطل",
    "بطلة",
    "بطولة",
    "متصدر",
    "متصدرة",
    "تفوق رياضي",
    "لياقة",
    "قوة بدنية",
    "انتصار",
    # Health / wellbeing (neutral-positive, general wellness only)
    "بصحة جيدة",
    "لياقة عالية",
    "نشيط",
    "نشيطة",
    "حيوية",
    "طاقة ايجابية",
    "راحة البال",
    "استرخاء",
    "هدوء",
    # Travel
    "اجازة ممتعة",
    "إجازة ممتعة",
    "رحلة رائعة",
    "منظر خلاب",
    "اطلالة رائعة",
    "إطلالة رائعة",
    "ضيافة",
    "كرم",
    "استقبال حار",
    # Education
    "تفوق دراسي",
    "تفوق أكاديمي",
    "درجة عالية",
    "ترتيب متقدم",
    "تخرج",
    "تخرجت",
    "منحة",
    "قبول",
    "اجتياز",
    "اجتزت",
}

_NEGATIVE = {
    # Core quality complaints
    "سيء",
    "سيئة",
    "رديء",
    "رديئة",
    "زفت",
    "فاشل",
    "فاشلة",
    "فشل",
    "ضعيف",
    "ضعيفة",
    "بطيء",
    "بطيئة",
    "معطل",
    "معطلة",
    "قديم",
    "قديمة",
    "متهالك",
    "متهالكة",
    "وسخ",
    "وسخة",
    "مقرف",
    "مقرفة",
    "بشع",
    "بشعة",
    "خايس",
    "خايسة",
    "تعبان",
    "تعبانة",
    "مكسور",
    "مكسورة",
    # Emotion / reaction
    "حزين",
    "حزينة",
    "زعلان",
    "زعلانة",
    "غاضب",
    "غاضبة",
    "متضايق",
    "متضايقة",
    "مستاء",
    "مستاءة",
    "محبط",
    "محبطة",
    "تعيس",
    "تعيسة",
    "خايف",
    "خايفة",
    "قلقان",
    "قلقانة",
    "متنرفز",
    "متنرفزة",
    "طفشان",
    "طفشانة",
    "مصدوم",
    "مصدومة",
    "مقهور",
    "مقهورة",
    # Complaints / outcome
    "اكره",
    "أكره",
    "بكره",
    "كارثة",
    "خيبة",
    "خيبة امل",
    "خيبة أمل",
    "مخيب",
    "مخيبة",
    "مشكلة",
    "مشاكل",
    "عيب",
    "عيوب",
    "غش",
    "نصب",
    "استغلال",
    "احتيال",
    "خسارة",
    "خسرت",
    "ندمت",
    "نادم",
    "نادمة",
    "للاسف",
    "للأسف",
    "يا خسارة",
    "يا للاسف",
    # Service/product specific (reviews)
    "سيئة جدا",
    "سيء جدا",
    "لا انصح",
    "لا انصح به",
    "لا يستاهل",
    "مضيعة وقت",
    "مضيعة فلوس",
    "ما يستاهل",
    "اسوأ",
    "أسوأ",
    "اسوأ تجربة",
    "أسوأ تجربة",
    "خدمة سيئة",
    "خدمة زفت",
    "تعامل سيء",
    "رد فعل سيء",
    "مزعج",
    "مزعجة",
    "ممل",
    "ممله",
    "مملة",
    "معقد",
    "معقدة",
    "غير مفهوم",
    "بطء",
    "تأخير",
    "تاخير",
    "تأخر",
    "تاخر",
    # Work / business
    "غير منتج",
    "غير منتجة",
    "مهمل",
    "مهملة",
    "غير كفوء",
    "غير كفوءة",
    "فوضوي",
    "فوضوية",
    "متسيب",
    "متسيبة",
    "غير موثوق",
    "غير موثوقة",
    "غير محترف",
    "غير محترفة",
    "خسارة مالية",
    "خسائر",
    "فصل",
    "فصلوه",
    "استقالة",
    "انهيار",
    "افلاس",
    "إفلاس",
    "تراجع",
    "ركود",
    "تسريح",
    "تسريح عمال",
    "اقالة",
    "إقالة",
    # Sports / competition
    "خسارة المباراة",
    "خسر",
    "خسروا",
    "هزيمة",
    "انهزم",
    "انهزموا",
    "اقصاء",
    "إقصاء",
    "استبعاد",
    "طرد",
    "بطاقة حمراء",
    "تراجع في الترتيب",
    # Health / wellbeing (general, non-diagnostic)
    "متعب",
    "متعبة",
    "منهك",
    "منهكة",
    "ارهاق",
    "إرهاق",
    "تعب شديد",
    "قلة نوم",
    "وعكة",
    "غير مرتاح",
    "غير مرتاحة",
    # Travel
    "رحلة سيئة",
    "تأخير الرحلة",
    "تاخير الرحلة",
    "الغاء الرحلة",
    "إلغاء الرحلة",
    "فقدان الأمتعة",
    "فقدان الامتعة",
    "خدمة سيئة في الفندق",
    "استقبال بارد",
    "سوء ضيافة",
    # Education
    "رسوب",
    "رسب",
    "رسبت",
    "راسب",
    "راسبة",
    "درجة منخفضة",
    "تراجع دراسي",
    "رفض القبول",
    "رفض",
    "فصل من الجامعة",
    "انذار اكاديمي",
    "إنذار أكاديمي",
}

_NEGATION = {"مش", "ما", "لا", "مو", "مب", "ماب", "لم", "لن", "ليس"}

_TOKEN_RE = re.compile(r"[؀-ۿ]+")

# Split each lexicon into single words (matched token-by-token, negation-aware)
# and multi-word phrases (matched as substrings against the raw text - things
# like "ما شاء الله" or "خدمة سيئة" don't survive tokenization as one unit).
_POSITIVE_WORDS = {w for w in _POSITIVE if " " not in w}
_POSITIVE_PHRASES = sorted((w for w in _POSITIVE if " " in w), key=len, reverse=True)
_NEGATIVE_WORDS = {w for w in _NEGATIVE if " " not in w}
_NEGATIVE_PHRASES = sorted((w for w in _NEGATIVE if " " in w), key=len, reverse=True)


def _strip_clitics(token: str) -> str:
    """Strip common single-letter conjunction/preposition prefixes (و ف ب ل)
    so 'وممتاز' still matches 'ممتاز' in the lexicon. Deliberately narrow -
    only strips one leading clitic, never touches the definite article
    'ال' since that's part of many lexicon entries as-is."""
    if len(token) > 3 and token[0] in "وفبل":
        return token[1:]
    return token


def _lexicon_sentiment(text: str, fallback_reason: str | None = None) -> SentimentResult:
    pos_hits, neg_hits = [], []

    # Phrase pass: longest phrases first so a longer match "wins" over a
    # shorter one it contains (e.g. "اسوأ تجربة" before "اسوأ" alone -
    # not perfect, still a heuristic, but reduces obvious double-counts).
    remaining_text = text
    for phrase in _POSITIVE_PHRASES:
        if phrase in remaining_text:
            pos_hits.append(phrase)
            remaining_text = remaining_text.replace(phrase, " ", 1)
    for phrase in _NEGATIVE_PHRASES:
        if phrase in remaining_text:
            neg_hits.append(phrase)
            remaining_text = remaining_text.replace(phrase, " ", 1)

    # Token pass: single words, negation-aware.
    tokens = [_strip_clitics(t) for t in _TOKEN_RE.findall(remaining_text)]
    for i, tok in enumerate(tokens):
        preceded_by_negation = i > 0 and tokens[i - 1] in _NEGATION
        if tok in _POSITIVE_WORDS:
            (neg_hits if preceded_by_negation else pos_hits).append(tok)
        elif tok in _NEGATIVE_WORDS:
            (pos_hits if preceded_by_negation else neg_hits).append(tok)

    score = len(pos_hits) - len(neg_hits)
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    bounded_score = max(-1.0, min(1.0, score / 4))
    return SentimentResult(
        text=text,
        label=cast(SentimentLabel, label),
        score=bounded_score,
        method="negation-aware-lexicon-v2",
        matched_positive=pos_hits,
        matched_negative=neg_hits,
        fallback_reason=fallback_reason,
        warnings=["Lexicon score is heuristic, not a calibrated probability."],
    )


def analyze_sentiment(text: str, *, backend: str = "auto") -> SentimentResult:
    """Classify `text` as positive/negative/neutral.

    Tries the CAMeL-Lab Arabic sentiment model first; transparently falls
    back to a negation-aware lexicon scorer if the model isn't reachable.
    `result["method"]` always says which path actually ran.
    """
    text = validate_text(text)
    if backend not in {"auto", "transformer", "lexicon"}:
        raise ValueError("backend must be one of: auto, transformer, lexicon")
    if backend == "lexicon":
        return _lexicon_sentiment(text)

    pipe = _get_pipeline()
    if pipe is not None:
        try:
            out = pipe(text, truncation=True)[0]
            label = out["label"].lower()
            if label not in {"positive", "negative", "neutral"}:
                raise ValueError(f"unsupported model label: {label}")
            probability = round(float(out["score"]), 3)
            signed = (
                probability if label == "positive" else -probability if label == "negative" else 0.0
            )
            return SentimentResult(
                text=text,
                label=cast(SentimentLabel, label),
                score=signed,
                method=f"transformer-model:{_MODEL_ID}",
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            if backend == "transformer":
                raise RuntimeError("transformer inference failed") from exc
            return _lexicon_sentiment(text, f"inference_failed:{type(exc).__name__}")

    if backend == "transformer":
        raise RuntimeError(
            f"transformer backend unavailable: {_pipeline_failure_reason or 'not available'}"
        )
    return _lexicon_sentiment(text, _pipeline_failure_reason or "transformer_unavailable")

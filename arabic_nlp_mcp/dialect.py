"""Arabic dialect identification.

Honest design note: there is no reliably available, zero-setup pretrained
dialect-ID model to call here - the well-known research models (e.g. NADI
shared-task systems) need weights hosted on the Hub, gated behind normal
internet access this sandbox doesn't have while building this package.
Rather than claim a model that can't be verified, this ships as a
transparent lexicon/marker-word classifier: it scores text against curated
marker-word lists for four buckets (Gulf, Egyptian, Levantine, MSA) and
returns inspectable evidence per bucket, not a probability.

This is a legitimate, if less accurate, approach - it's how a lot of
production dialect-routing actually starts before a labeled dataset exists
to train a classifier on. The method is reported in the result so callers
always know they're getting a heuristic, not a black box.
"""

from __future__ import annotations

from collections import Counter
from typing import cast

from .common import ARABIC_TOKEN_RE, validate_text
from .models import DialectLabel, DialectResult

DIALECTS = ("gulf", "egyptian", "levantine", "msa")

# Curated marker words/particles per dialect. Not exhaustive - this is a
# starting lexicon meant to be extended (see README for how to add terms).
_MARKERS = {
    "gulf": [
        # Question words / particles
        "وش",
        "شلون",
        "شنو",
        "ليش",
        "وين",
        "متى",
        "كم",
        "ايش",
        "اشلون",
        "شخبارك",
        "شخبارج",
        "شحالك",
        "شسالفة",
        "شفيه",
        # Want / go / come / pronouns
        "ابغى",
        "أبغى",
        "ابي",
        "أبي",
        "ابغاه",
        "يبيله",
        "يبيه",
        "تبي",
        "تبغى",
        "ودي",
        "وياك",
        "وياج",
        "وياكم",
        "تراني",
        "تراك",
        "تراه",
        "ماكو",
        "اكو",
        "ماعندي",
        "عندك",
        # Negation / existence (Gulf-specific forms)
        "مب",
        "ماب",
        "موب",
        "مو صح",
        "ماهو",
        # Time / discourse particles
        "الحين",
        "توه",
        "توني",
        "لين",
        "قبل شوي",
        "عقب",
        "بعدين",
        "عاد",
        "خلاص",
        "شوي",
        "اشوي",
        "مره",
        "وايد",
        "زين",
        "خوش",
        "هني",
        "هنا",
        "چذي",
        "كذا كذا",
        "يعني كذا",
        # Common Gulf/Saudi expressions
        "يا هلا",
        "حياك",
        "حياك الله",
        "الله يحييك",
        "تسلم",
        "يسلمو",
        "ماشاء الله",
        "قوة",
        "طاح",
        "دشيت",
        "دشرت",
        "سويت",
        "سالفة",
        "جذي",
        "بعدين نشوف",
    ],
    "egyptian": [
        # Question words / particles
        "ايه",
        "إيه",
        "فين",
        "ازاي",
        "إزاي",
        "ليه",
        "امتى",
        "إمتى",
        "قد ايه",
        "مين",
        # Want / pronouns / verbs
        "عايز",
        "عايزة",
        "عاوز",
        "عاوزة",
        "هروح",
        "هعمل",
        "هاجي",
        "جاي",
        "رايح",
        "ماشي",
        "ماشية",
        # Negation
        "مش",
        "مافيش",
        "معلهش",
        "معلش",
        # Discourse / intensifiers
        "كده",
        "كدة",
        "خالص",
        "دلوقتي",
        "عشان",
        "قوي",
        "اوي",
        "أوي",
        "اهو",
        "لسه",
        "طب",
        "بجد",
        "خلاص كده",
        "يلا بينا",
        "بص",
        "بصراحة",
        # Common Egyptian expressions
        "ازيك",
        "إزيك",
        "عامل ايه",
        "تمام كده",
        "جامد",
        "حلوين",
        "قشطة",
        "تحفة",
        "فل",
        "زي الفل",
        "يا عم",
        "يا باشا",
        "استنى",
        "هو ايه ده",
    ],
    "levantine": [
        # Question words / particles
        "شو",
        "وين",
        "ليش",
        "كيفك",
        "قديش",
        "امتى",
        "إمتى",
        "مين",
        "لوين",
        # Want / pronouns / verbs
        "بدي",
        "بدك",
        "بدو",
        "بدها",
        "بدنا",
        "رح",
        "رايح",
        "جاي",
        # Negation
        "مو",
        "ما في",
        "منيح مو",
        # Discourse / intensifiers
        "هيك",
        "هلق",
        "هلأ",
        "كتير",
        "شوي",
        "كمان",
        "منيح",
        "منيحة",
        "تمام",
        "يلا",
        "ولو",
        "مبين",
        "معقول",
        "عنجد",
        "صرلي",
        "يعطيك العافية",
        # Common Levantine expressions
        "شلونك",
        "كيف الحال",
        "الله يعطيك العافية",
        "تسلملي",
        "خيو",
        "زلمة",
        "يا زلمة",
        "ماشي الحال",
        "عالماشي",
    ],
    "msa": [
        "إن",
        "لكن",
        "حيث",
        "بينما",
        "إذ",
        "لأن",
        "غير أن",
        "بيد أن",
        "وبالتالي",
        "علاوة على ذلك",
        "إذا",
        "عندما",
        "الذي",
        "التي",
        "الذين",
        "اللواتي",
        "إلا أن",
        "من أجل",
        "بغية",
        "نظرا لـ",
        "نظراً لـ",
        "وعليه",
        "وفي هذا السياق",
        "جدير بالذكر",
        "تجدر الإشارة",
        "بناء على ذلك",
        "بناءً على ذلك",
        "فيما يتعلق بـ",
        "من ثم",
        "وعلى الرغم من",
        "رغم ذلك",
        "في حين أن",
        "مما لا شك فيه",
    ],
}


def _normalized_tokens(text: str) -> list[str]:
    return [t.strip("ًٌٍَُِّْٰ") for t in ARABIC_TOKEN_RE.findall(text)]


def _contains_marker(tokens: list[str], marker: str) -> bool:
    wanted = marker.split()
    if not wanted or len(wanted) > len(tokens):
        return False
    return any(tokens[i : i + len(wanted)] == wanted for i in range(len(tokens) - len(wanted) + 1))


def detect_dialect(text: str) -> DialectResult:
    """Score `text` against marker-word lists for four dialect buckets.

    Returns normalized evidence shares plus a label. With no evidence it
    returns unknown; close competing buckets return mixed.
    """
    text = validate_text(text)
    tokens = _normalized_tokens(text)
    ownership = Counter(marker for markers in _MARKERS.values() for marker in set(markers))
    raw_scores: dict[str, float] = {}
    matched_terms = {}
    for dialect, markers in _MARKERS.items():
        hits = [m for m in markers if _contains_marker(tokens, m)]
        # Phrases carry more evidence; markers shared by dialects carry less.
        raw_scores[dialect] = sum((1.5 if " " in m else 1.0) / ownership[m] for m in hits)
        matched_terms[dialect] = hits

    total = sum(raw_scores.values())
    if total == 0:
        return DialectResult(
            text=text,
            predicted="unknown",
            evidence_score=0.0,
            scores={d: 0.0 for d in DIALECTS},
            matched_terms=matched_terms,
            warnings=["No known dialect evidence was found."],
        )

    scores = {d: round(raw_scores[d] / total, 3) for d in DIALECTS}
    ranked = sorted(raw_scores, key=lambda dialect: raw_scores[dialect], reverse=True)
    top, second = ranked[:2]
    top_score, second_score = raw_scores[top], raw_scores[second]
    predicted = cast(
        DialectLabel, "mixed" if second_score > 0 and top_score / second_score < 1.5 else top
    )
    warnings = ["Heuristic evidence score; it is not a calibrated probability."]
    if predicted == "mixed":
        warnings.append("Evidence is split across dialect buckets.")
    return DialectResult(
        text=text,
        predicted=predicted,
        evidence_score=scores[top],
        scores=scores,
        matched_terms=matched_terms,
        warnings=warnings,
    )

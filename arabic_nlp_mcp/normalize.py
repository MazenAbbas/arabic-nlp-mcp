"""Arabic text normalization.

Pure rule-based, deterministic, Unicode-native, and dependency-light.
"""

from __future__ import annotations

import re

from .common import CONTROL_CHAR_RE, FORMAT_CONTROL_RE, validate_text
from .models import NormalizeResult

# Common "informal" letter variants people type instead of the correct form.
_ALEF_VARIANTS = re.compile("[إأآا]")
_YEH_VARIANTS = re.compile("[يى]")
_TEH_MARBUTA = re.compile("[ة]")
_TATWEEL = re.compile("[ـ]+")
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_REPEATED_CHAR = re.compile(r"([\u0621-\u063A\u0641-\u064A])\1{2,}")


def normalize(
    text: str,
    *,
    strip_diacritics: bool = True,
    unify_alef: bool = True,
    unify_yeh: bool = True,
    unify_teh_marbuta: bool = False,
    remove_tatweel: bool = True,
    remove_control_characters: bool = True,
    collapse_repeated_letters: bool = True,
) -> NormalizeResult:
    """Normalize Arabic text for downstream NLP (search, matching, dedup).

    Each step is individually toggleable because different use cases want
    different normalization: search indexing usually wants everything on,
    while a diacritization pipeline wants strip_diacritics=False so it
    doesn't destroy input it's supposed to restore.

    Returns a dict with the normalized text plus a log of which
    transformations actually changed something (useful for debugging and
    for showing your work in a demo).
    """
    text = validate_text(text)

    original = text
    applied = []

    if remove_control_characters:
        new = CONTROL_CHAR_RE.sub("", FORMAT_CONTROL_RE.sub("", text))
        if new != text:
            applied.append("remove_control_characters")
        text = new

    if remove_tatweel:
        new = _TATWEEL.sub("", text)
        if new != text:
            applied.append("remove_tatweel")
        text = new

    if strip_diacritics:
        new = _DIACRITICS.sub("", text)
        if new != text:
            applied.append("strip_diacritics")
        text = new

    if unify_alef:
        new = _ALEF_VARIANTS.sub("ا", text)
        if new != text:
            applied.append("unify_alef")
        text = new

    if unify_yeh:
        new = _YEH_VARIANTS.sub("ي", text)
        if new != text:
            applied.append("unify_yeh")
        text = new

    if unify_teh_marbuta:
        new = _TEH_MARBUTA.sub("ه", text)
        if new != text:
            applied.append("unify_teh_marbuta")
        text = new

    if collapse_repeated_letters:
        new = _REPEATED_CHAR.sub(r"\1\1", text)  # keep max 2 repeats
        if new != text:
            applied.append("collapse_repeated_letters")
        text = new

    # Collapse whitespace last, always.
    new = re.sub(r"\s+", " ", text).strip()
    if new != text:
        applied.append("collapse_whitespace")
    text = new

    lossy_steps = {
        "strip_diacritics",
        "unify_alef",
        "unify_yeh",
        "unify_teh_marbuta",
        "collapse_repeated_letters",
    }
    lossy = any(step in lossy_steps for step in applied)
    warnings = ["Normalization removed distinctions; retain the original text."] if lossy else []
    return NormalizeResult(
        original=original,
        normalized=text,
        changed=text != original,
        lossy=lossy,
        transformations_applied=applied,
        warnings=warnings,
    )

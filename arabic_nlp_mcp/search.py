"""Arabic-aware preprocessing for search, RAG, and deterministic deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from typing import Literal

from .common import CONTROL_CHAR_RE, FORMAT_CONTROL_RE, validate_text
from .models import SearchPreparationResult

Profile = Literal["conservative", "search", "aggressive"]

_ARABIC_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640+")
_ALEF = re.compile("[إأآٱ]")
_YEH = re.compile("[ىیي]")
_KAF = re.compile("[كک]")
_HAMZA = re.compile("[ؤئ]")
_ARABIC_REPEAT = re.compile(r"([\u0621-\u063A\u0641-\u064A])\1{2,}")
_TOKEN = re.compile(r"[\u0621-\u063A\u0641-\u064A0-9]+|[A-Za-z]+(?:['’-][A-Za-z]+)*")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PUNCTUATION = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ","})


def _step(text: str, name: str, operation: Callable[[str], str], applied: list[str]) -> str:
    changed = operation(text)
    if changed != text:
        applied.append(name)
    return changed


def prepare_for_search(text: str, *, profile: Profile = "search") -> SearchPreparationResult:
    """Create a stable search form without mutating the original text.

    conservative removes invisible noise and canonicalizes compatibility
    forms; search also removes Arabic marks and unifies common letter/digit
    variants; aggressive additionally collapses expressive letter repetition
    and maps taa marbuta for maximum recall at a higher false-match risk.
    """
    if profile not in {"conservative", "search", "aggressive"}:
        raise ValueError("profile must be one of: conservative, search, aggressive")
    original = validate_text(text)
    value = original
    applied: list[str] = []
    warnings: list[str] = []

    value = _step(value, "unicode_nfkc", lambda s: unicodedata.normalize("NFKC", s), applied)
    value = _step(value, "remove_control_characters", lambda s: CONTROL_CHAR_RE.sub("", s), applied)
    value = _step(value, "remove_format_controls", lambda s: FORMAT_CONTROL_RE.sub("", s), applied)
    value = _step(value, "remove_tatweel", lambda s: _TATWEEL.sub("", s), applied)

    if profile in {"search", "aggressive"}:
        value = _step(value, "strip_arabic_marks", lambda s: _ARABIC_MARKS.sub("", s), applied)
        value = _step(value, "unify_alef", lambda s: _ALEF.sub("ا", s), applied)
        value = _step(value, "unify_yeh", lambda s: _YEH.sub("ي", s), applied)
        value = _step(value, "unify_kaf", lambda s: _KAF.sub("ك", s), applied)
        value = _step(value, "normalize_digits", lambda s: s.translate(_DIGITS), applied)
        value = _step(value, "normalize_punctuation", lambda s: s.translate(_PUNCTUATION), applied)
        warnings.append(
            "Search normalization is lossy; retain the original text for display and audit."
        )

    if profile == "aggressive":
        value = _step(value, "unify_hamza_seats", lambda s: _HAMZA.sub("ء", s), applied)
        value = _step(value, "unify_teh_marbuta", lambda s: s.replace("ة", "ه"), applied)
        value = _step(
            value, "collapse_repeated_letters", lambda s: _ARABIC_REPEAT.sub(r"\1\1", s), applied
        )
        warnings.append("Aggressive profile improves recall but may merge distinct Arabic words.")

    value = _step(value, "casefold_latin", lambda s: s.casefold(), applied)
    value = _step(value, "collapse_whitespace", lambda s: re.sub(r"\s+", " ", s).strip(), applied)
    tokens = _TOKEN.findall(value)
    fingerprint_source = "\x1f".join(tokens).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_source).hexdigest()
    return SearchPreparationResult(
        original=original,
        search_text=value,
        tokens=tokens,
        fingerprint=fingerprint,
        changed=value != original,
        lossy=profile != "conservative",
        transformations_applied=applied,
        profile=profile,
        warnings=warnings,
    )


def prepare_batch(
    texts: Iterable[str], *, profile: Profile = "search"
) -> Iterator[SearchPreparationResult]:
    """Lazily prepare a stream of texts without holding the corpus in memory."""
    for text in texts:
        yield prepare_for_search(text, profile=profile)

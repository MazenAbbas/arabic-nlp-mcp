"""Shared validation and Unicode helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated

from pydantic import Field

MAX_TEXT_LENGTH = 20_000
BoundedText = Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]
ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A]")
ARABIC_TOKEN_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u064B-\u065F\u0670\u0671-\u06D3]+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
FORMAT_CONTROL_RE = re.compile(r"[\u061C\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]")


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text must contain at most {MAX_TEXT_LENGTH} characters")
    return unicodedata.normalize("NFC", text)

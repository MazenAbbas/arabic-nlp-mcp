"""Stable structured result contracts for public tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DialectLabel = Literal["gulf", "egyptian", "levantine", "msa", "mixed", "unknown"]
SentimentLabel = Literal["positive", "negative", "neutral"]


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizeResult(ResultModel):
    original: str
    normalized: str
    changed: bool
    lossy: bool
    transformations_applied: list[str]
    method: Literal["unicode-rules-v2"] = "unicode-rules-v2"
    warnings: list[str] = Field(default_factory=list)


class SearchPreparationResult(ResultModel):
    original: str
    search_text: str
    tokens: list[str]
    fingerprint: str
    changed: bool
    lossy: bool
    transformations_applied: list[str]
    profile: Literal["conservative", "search", "aggressive"]
    method: Literal["arabic-search-preparation-v1"] = "arabic-search-preparation-v1"
    warnings: list[str] = Field(default_factory=list)


class DialectResult(ResultModel):
    text: str
    predicted: DialectLabel
    evidence_score: float = Field(ge=0, le=1)
    scores: dict[str, float]
    matched_terms: dict[str, list[str]]
    method: Literal["weighted-marker-heuristic-v2"] = "weighted-marker-heuristic-v2"
    warnings: list[str] = Field(default_factory=list)


class SentimentResult(ResultModel):
    text: str
    label: SentimentLabel
    score: float = Field(ge=-1, le=1)
    method: str
    matched_positive: list[str] = Field(default_factory=list)
    matched_negative: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class WordReport(ResultModel):
    word: str
    diacritized: str
    in_dictionary: bool
    matched_via: Literal["direct", "al_stripped"] | None = None


class DiacritizeResult(ResultModel):
    original: str
    diacritized: str
    coverage: float = Field(ge=0, le=1)
    words_covered: int = Field(ge=0)
    words_total: int = Field(ge=0)
    word_report: list[WordReport]
    method: Literal["lexical-dictionary-v2"] = "lexical-dictionary-v2"
    warnings: list[str] = Field(default_factory=list)

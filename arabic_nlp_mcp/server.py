"""Stable and opt-in experimental MCP server entry points."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .common import BoundedText
from .diacritize import diacritize
from .dialect import detect_dialect
from .models import (
    DiacritizeResult,
    DialectResult,
    NormalizeResult,
    SearchPreparationResult,
    SentimentResult,
)
from .normalize import normalize
from .search import Profile, prepare_for_search
from .sentiment import analyze_sentiment

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def arabic_normalize(
    text: BoundedText,
    strip_diacritics: bool = True,
    unify_alef: bool = True,
    unify_yeh: bool = True,
    unify_teh_marbuta: bool = False,
    remove_tatweel: bool = True,
    remove_control_characters: bool = True,
    collapse_repeated_letters: bool = True,
) -> NormalizeResult:
    """Normalize Arabic text with individually controlled Unicode rules."""
    return normalize(
        text,
        strip_diacritics=strip_diacritics,
        unify_alef=unify_alef,
        unify_yeh=unify_yeh,
        unify_teh_marbuta=unify_teh_marbuta,
        remove_tatweel=remove_tatweel,
        remove_control_characters=remove_control_characters,
        collapse_repeated_letters=collapse_repeated_letters,
    )


def arabic_prepare_for_search(
    text: BoundedText, profile: Profile = "search"
) -> SearchPreparationResult:
    """Prepare Arabic/mixed text for search, RAG, and deduplication.

    Returns normalized tokens, a stable SHA-256 fingerprint, lossiness,
    warnings, and the exact transformation audit trail.
    """
    return prepare_for_search(text, profile=profile)


def arabic_detect_dialect(text: BoundedText) -> DialectResult:
    """Experimental weighted marker evidence; not a calibrated classifier."""
    return detect_dialect(text)


def arabic_sentiment(text: BoundedText, backend: str = "auto") -> SentimentResult:
    """Experimental sentiment with explicit model/lexicon backend behavior."""
    return analyze_sentiment(text, backend=backend)


def arabic_diacritize(text: BoundedText) -> DiacritizeResult:
    """Experimental lexical lookup; does not infer grammatical case."""
    return diacritize(text)


def _build_server(*, include_experimental: bool) -> MCPServer:
    qualifier = " with opt-in experimental linguistic tools" if include_experimental else ""
    server = MCPServer(
        name="arabic-nlp-mcp",
        version="0.4.0rc1",
        instructions=(
            "Dependable Arabic preprocessing for search, RAG, and deduplication"
            f"{qualifier}. Preserve original text and inspect lossiness/warnings."
        ),
    )
    server.tool(annotations=READ_ONLY)(arabic_normalize)
    server.tool(annotations=READ_ONLY)(arabic_prepare_for_search)
    if include_experimental:
        server.tool(annotations=READ_ONLY)(arabic_detect_dialect)
        server.tool(annotations=READ_ONLY)(arabic_sentiment)
        server.tool(annotations=READ_ONLY)(arabic_diacritize)
    return server


mcp = _build_server(include_experimental=False)
experimental_mcp = _build_server(include_experimental=True)


def main() -> None:
    mcp.run()


def main_experimental() -> None:
    experimental_mcp.run()


if __name__ == "__main__":
    main()

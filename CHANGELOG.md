# Changelog

## 0.4.0rc1 - 2026-09-15

- Added a production-focused search/RAG/deduplication preparation pipeline.
- Added conservative, search, and aggressive profiles with explicit lossiness.
- Added mixed Arabic/Latin tokenization and deterministic SHA-256 fingerprints.
- Added lazy batch processing and UTF-8 JSONL CLI support.
- Classified dialect, sentiment, and lexical diacritization as experimental.
- Removed unsafe control and bidi-format characters from stable processed output.
- Added Persian kaf/yeh interoperability for search normalization.
- Added property-based Unicode fuzzing and real MCP stdio integration tests.
- Added strict typing, live dependency audit, Dependabot, and a threat model.

## 0.3.0 - 2026-09-15

- Migrated to the MCP Python SDK v2 and typed Pydantic output contracts.
- Added read-only, idempotent, non-destructive, closed-world annotations.
- Added bounded input validation and Unicode-native normalization.
- Added exact token/phrase dialect matching plus `unknown` and `mixed` states.
- Made transformer sentiment optional and fallback behavior explicit.
- Reworked diacritization as lexical citation-form lookup with explicit
  feminine forms and sun-letter handling.
- Added release gates, regression tests, security policy, and honest limits.

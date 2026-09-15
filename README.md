# Arabic NLP MCP

An MCP v2 server centered on dependable Arabic preprocessing for search,
RAG, and deduplication, with additional experimental linguistic tools. It uses typed,
closed-world structured outputs. Results identify their method and expose
ambiguity, fallback, or coverage rather than presenting heuristic scores as
probabilities.

## Tools

| Tool | Behavior | Important boundary |
|---|---|---|
| `arabic_normalize` | Deterministic Unicode normalization | Options may be lossy; choose them for the downstream task |
| `arabic_prepare_for_search` | Profiles, mixed-language tokens, audit trail, and stable fingerprint | Primary supported workflow; `search` and `aggressive` profiles are explicitly lossy |
| `arabic_detect_dialect` | Weighted token/phrase evidence for Gulf, Egyptian, Levantine, and MSA | Heuristic; returns `unknown` or `mixed` when evidence is absent/ambiguous |
| `arabic_sentiment` | Explicit `auto`, `transformer`, or `lexicon` backend | Lexicon scores are not calibrated; transformer support is optional |
| `arabic_diacritize` | Lexical dictionary lookup with coverage report | Does not infer contextual grammar or case endings |

The default MCP server exposes only the two stable preprocessing tools. The
three linguistic tools are available through the deliberately named
`arabic-nlp-mcp-experimental` entry point, so clients cannot mistake them for
validated classifiers. All tools reject inputs over 20,000 characters and are annotated read-only,
idempotent, non-destructive, and closed-world in MCP.

## Install and run

Python 3.10 or newer is required.

```bash
python -m pip install .
arabic-nlp-mcp
```

To opt into the experimental dialect, sentiment, and lexical-diacritization
tools, configure the command as `arabic-nlp-mcp-experimental`.

Transformer sentiment is intentionally not part of the lightweight core:

```bash
python -m pip install ".[transformers]"
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "arabic-nlp": {
      "command": "python",
      "args": ["-m", "arabic_nlp_mcp.server"]
    }
  }
}
```

## Python API

Functions return Pydantic models. Serialize with `model_dump()` when a plain
dictionary is needed.

```python
from arabic_nlp_mcp.dialect import detect_dialect
from arabic_nlp_mcp.search import prepare_for_search
from arabic_nlp_mcp.sentiment import analyze_sentiment

prepare_for_search("إِنَّ  AI رقم ١٢٣").model_dump()
detect_dialect("وين").model_dump()  # predicted: mixed
analyze_sentiment("مش حلو", backend="lexicon").model_dump()
```

`backend="auto"` attempts the transformer and reports `fallback_reason` if it
uses the lexicon. `backend="transformer"` raises an error when unavailable; it
never silently falls back.

### Search profiles

- `conservative`: compatibility normalization and invisible-format cleanup;
  preserves Arabic marks, letter variants, and digits.
- `search`: additionally strips marks and unifies alef, yeh, Arabic/Persian
  digits, punctuation, and Latin case. This is the recommended default.
- `aggressive`: additionally merges hamza seats and taa marbuta and collapses
  expressive repetition. Use only when recall matters more than false matches.

The original text is always returned. `fingerprint` is SHA-256 over normalized
tokens, so equivalent search forms can be deduplicated without storing the
normalized content as an identifier.

For streaming files without loading the corpus into memory:

```bash
arabic-nlp search --jsonl < input.txt > prepared.jsonl
```

## Verification

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
python -m twine check dist/*
```

The test corpus is a regression suite, not an external accuracy benchmark.
This project publishes no accuracy, F1, or coverage claim until a licensed,
versioned external dataset and reproducible evaluation are added.

## Limitations

- Dialect identification covers only four broad buckets and code-switching is
  not modeled. The evidence score is descriptive, not calibrated confidence.
- Lexicon sentiment has limited vocabulary and basic local negation. The
  optional model has its own training-domain limitations.
- Diacritization is useful only where dictionary coverage is adequate. It is
  not a replacement for a contextual morphological model.
- Normalization can intentionally remove distinctions such as `ى` versus `ي`.

Do not use these heuristic outputs alone for high-stakes decisions.

`arabic_prepare_for_search` and `arabic_normalize` are the stable core.
Dialect, sentiment, and lexical diacritization are labeled experimental until
versioned external evaluations justify stronger guarantees.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[THREAT_MODEL.md](docs/THREAT_MODEL.md). Changes are recorded in
[CHANGELOG.md](CHANGELOG.md). Released under the MIT License.

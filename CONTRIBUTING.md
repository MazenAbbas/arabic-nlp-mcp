# Contributing

Use Python 3.10+ and install `.[dev]`. Before opening a pull request, run the
commands in the README verification section. Add regression tests for behavior
changes and preserve the typed result contracts unless the change is released
as a documented breaking version.

New linguistic rules need examples that demonstrate both the intended match
and a nearby false-positive case. Do not describe regression cases as an
accuracy benchmark. Model or dataset additions must document source, version,
license, evaluation method, and reproducibility steps.

Please keep stdout reserved for MCP protocol traffic; diagnostics belong on
stderr. Never add silent fallback behavior or call a heuristic score a
probability.

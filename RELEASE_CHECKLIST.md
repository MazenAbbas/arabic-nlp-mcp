# Release checklist

- [ ] Version and changelog agree.
- [ ] `ruff check .` passes.
- [ ] Strict `mypy arabic_nlp_mcp` passes.
- [ ] Full test suite passes on supported Python versions.
- [ ] Wheel and sdist build from a clean tree.
- [ ] `twine check` passes for both artifacts.
- [ ] Wheel installs in a new environment and `pip check` passes.
- [ ] MCP tool listing, annotations, schemas, and structured call pass.
- [ ] Real stdio handshake and installed CLI smoke tests pass.
- [ ] Current dependency audit reports no known vulnerabilities.
- [ ] Archive excludes environments, caches, build outputs, and placeholders.
- [ ] README claims match reproducible evidence.

# Threat model

## Scope

The stable server transforms caller-supplied text and has no filesystem,
network, credential, shell, database, or mutation tools. Its intended trust
boundary is an MCP host invoking the local stdio process.

## Defenses

- Every tool is read-only, idempotent, non-destructive, and closed-world.
- Inputs are typed and limited to 20,000 Unicode code points before processing.
- Stable preprocessing removes C0/C1 controls, bidi overrides, zero-width
  format controls, BOM, and Arabic Letter Mark from processed output.
- Outputs are closed Pydantic models, and MCP publishes output schemas.
- The default server excludes unvalidated linguistic classifiers.
- The core does no network access. Model downloads exist only behind the
  separately installed and explicitly selected experimental backend.
- JSONL processing is streaming, so corpus size does not determine memory use.

## Residual risks

- Unicode confusables are not fully resolved; broad confusable folding can
  corrupt legitimate Arabic and is intentionally avoided.
- SHA-256 fingerprints are content-derived identifiers, not authentication or
  authorization tokens.
- A caller can create CPU load through repeated maximum-size requests. Process
  limits and MCP-host rate limits remain operator responsibilities.
- Supply-chain risk remains in Python and GitHub Actions dependencies. CI runs
  dependency audits and Dependabot monitors both ecosystems.
- Experimental linguistic results can be wrong and must not drive high-stakes
  decisions.

## Non-goals

The server does not detect prompt injection, decide whether text is safe, or
sanitize content for HTML/SQL/shell contexts. Callers must apply the escaping
required by the destination system.

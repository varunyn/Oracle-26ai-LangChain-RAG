# Oracle Knowledge MCP final verification

Verification date: **2026-09-03**. This report separates deterministic local
proof, gated live-provider proof, and unavailable outer-client proof.

## Dependency and profile facts

| Item | Value |
| --- | --- |
| FastMCP | 4.0.2 |
| MCP Python SDK | 2.1.1 |
| Negotiated protocol in proven profiles | 2026-07-28 |
| Codex CLI | 0.143.0 installed; outer handshake unverified |
| OpenWebUI | 0.11.0 target row; runtime unavailable |
| 2026-07-28 / MCP SDK v2 | Supported by the production dependency line |

## Deterministic proof

The real child-process STDIO and ASGI-backed Streamable HTTP profiles use the
same production server factory and deterministic providers. Both prove
initialization, schemas, all three tools, successful search, no hits, invalid
input, forbidden raw/unlisted keys, embedding failure, Oracle failure, reranker
fallback, and sanitized public errors. Their normalized snapshots are compared.

Commands and results:

```text
.venv/bin/python -m pytest tests/workflow_tests/test_oracle_knowledge_deployment_profiles.py -q
2 passed

.venv/bin/python -m pytest tests/workflow_tests/test_oracle_knowledge_protocol_compatibility.py -q
4 passed

.venv/bin/python -m pytest tests/unit_tests/test_oracle_knowledge_legacy_removal.py tests/unit_tests/test_settings.py -q
5 passed
```

The manual deterministic harness also produced successful STDIO and HTTP
reports with protocol `2026-07-28`, the exact three tools, `success`, `no_hits`,
`backend_error`, and a sanitized failure flag:

```bash
.venv/bin/python tests/run_oracle_knowledge_client_compatibility.py --transport stdio --report /tmp/oracle-knowledge-stdio.json
.venv/bin/python tests/run_oracle_knowledge_client_compatibility.py --transport streamable-http --report /tmp/oracle-knowledge-http.json
```

## Gated live Oracle/OCI proof

The live integration test is intentionally opt-in and never substitutes fake
providers:

```bash
RUN_ORACLE_KNOWLEDGE_LIVE=1 \
  .venv/bin/python -m pytest tests/integration_tests/test_oracle_knowledge_live.py -m integration -q
```

It requires a real OCI config/profile, compartment, DSN, wallet, embedding
model, configured application default collection, and reranker. Set
`ORACLE_KNOWLEDGE_LIVE_COLLECTION` only when the live test should use a
different collection. Missing configuration or unavailable services produce an
explicit pytest skip reason. When live prerequisites are present it verifies
stable document/chunk IDs, rank ordering, normalized source and citation
fields, retrieval scores, reranking scores, and applied reranking.
The supported container profile was executed with the repository's configured
Oracle and OCI services and `ORACLE_KNOWLEDGE_LIVE_QUERY='how to config oci cli in linux'`:

```text
1 passed
```

That run performed two real searches and proved stable document/chunk IDs,
ranked retrieval and reranking scores, applied OCI reranking, and canonical
citation normalization. A host-side run skipped with `readiness unavailable`
because the configured wallet and OCI paths are container mounts; this is an
expected deployment distinction, not provider proof.

With live credentials absent in the verification environment, the gated command
returned `1 skipped` with the explicit reason that `RUN_ORACLE_KNOWLEDGE_LIVE=1`
was not set.

## Manual repeated-search and citation smoke procedure

These steps apply only to the proven FastMCP Client deployment profiles and are
deterministic unless pointed at a live provider:

1. Start the STDIO server or Streamable HTTP profile documented in
   [ORACLE-KNOWLEDGE-MCP.md](../ORACLE-KNOWLEDGE-MCP.md).
2. Connect one real MCP client and verify protocol `2026-07-28` and the
   exact three tools.
3. Call `search_knowledge` twice in the same client session with the same
   friendly key, then once with a no-hit query. Confirm ordered evidence and
   stable document/chunk IDs.
4. The outer agent—not this server—writes the final answer. Render citations
   from each returned evidence item's normalized source/title/page/link fields.
5. Repeat the same sequence over the other transport and compare envelopes.

Codex CLI and OpenWebUI are not listed as supported because outer-client
initialization, repeated calls, and final citation rendering were not proven in
this environment. An ephemeral Codex CLI 0.143.0 Streamable HTTP run was
prepared against a ready temporary local server, but it was not authorized
because completing it would send private Oracle evidence to the configured
external Codex model. No global Codex configuration was changed. OpenWebUI
0.11.0 was not installed, so no OpenWebUI handshake or tool invocation was
attempted.

## Final repository and live workflow checks

```text
./scripts/regression_guard.sh
331 passed, 21 skipped; OpenAPI/docs drift and Agent Server SSE smoke passed

.venv/bin/python -m pytest <Oracle Knowledge + RAG/mixed focused suites> -q
119 passed

RUN_INTEGRATION_TESTS=1 LANGGRAPH_API_URL=http://127.0.0.1:2024 \
  .venv/bin/python -m pytest tests/integration_tests/test_langgraph_direct_and_rag_live.py -q
2 passed

uv run ruff check
passed

uv lock --check
passed

Node 24: cd docs-site && npm run build
passed
```

The changed Python files pass Black. The repository-wide Black check remains a
release blocker because 13 unrelated existing files require formatting.
`frontend/pnpm check` likewise reports 183 existing diagnostics, and
`frontend/pnpm build` reaches TypeScript then fails in unchanged
`src/hooks/chat/useChatController.ts` at the `NativeToolCall` to `Record`
conversion. The changed frontend projection tests pass (59 frontend tests in
the invoked suite). These unresolved checks must be cleared before tagging.

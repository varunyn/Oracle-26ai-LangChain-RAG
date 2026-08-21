# Oracle Knowledge MCP

Oracle Knowledge MCP is an evidence-retrieval server. It embeds a query,
retrieves Oracle Vector evidence, optionally reranks candidates, and returns
typed evidence and citations. It never generates chat answers; the caller owns
answer synthesis and citation presentation.

## Contract

The server exposes exactly three tools:

- `search_knowledge`: bounded vector retrieval and optional reranking.
- `list_knowledge_bases`: list configured friendly keys.
- `list_documents`: discover documents for a friendly key.

Public identifiers are friendly keys such as `default` or `handbook`. Raw table
or collection names are never accepted or returned. Search outcomes are typed:
`success`, `no_hits`, `invalid_request`, `forbidden`, or `backend_error`.

Configured limits are reported in server instructions and enforced by the
service: query length, result count, candidate count, and metadata filter count.
The default values are 8192, 50, 100, and 8 respectively. Reranking status is
`disabled`, `applied`, or `failed`.

## Prerequisites and configuration

Provide OCI config/profile, an embedding model, Oracle Vector credentials,
wallet, DSN, and a friendly mapping in `.env`:

```env
ORACLE_KNOWLEDGE_BASES={"default":"RAG_KNOWLEDGE_BASE"}
ORACLE_KNOWLEDGE_DEFAULT_KEY=default
ORACLE_KNOWLEDGE_TRANSPORT=stdio
ORACLE_KNOWLEDGE_HOST=127.0.0.1
ORACLE_KNOWLEDGE_PORT=9000
```

Only `stdio` and `streamable-http` are supported. SSE and other transports are
rejected. The standalone server uses only the namespaced transport settings.

## Run profiles

Run STDIO locally:

```bash
uv run python mcp_servers/oracle_knowledge.py
```

For HTTP, set the transport explicitly. The MCP endpoint is `/mcp`:

```bash
ORACLE_KNOWLEDGE_TRANSPORT=streamable-http \
  ORACLE_KNOWLEDGE_HOST=127.0.0.1 \
  uv run python mcp_servers/oracle_knowledge.py
```

The isolated deployment profile mounts OCI config, wallet, and key material
read-only:

```bash
docker compose -f docker-compose.oracle-knowledge.yml up --build
```

It binds HTTP to loopback by default and provides `/health/live` (process-only)
and `/health/ready` (configuration plus lightweight Oracle/OCI dependency
checks). Readiness failures return a fixed sanitized response.

## Security and observability

There is no built-in MCP authentication. Use loopback for local use or place
HTTP behind a trusted authenticated network boundary. Do not put credentials,
wallet paths, DSNs, SQL, raw provider payloads, query text, or document content
in logs or traces.

Set `ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING=true` to export traces from the
dedicated process. Its Compose profile routes optional traces/logs to
`host.docker.internal:4318`; no observability stack is required. The standalone
service name is `oracle-knowledge-mcp`; exporting is fail-open. Safe span fields include request ID, friendly key, configured limits,
query length, counts, timing, outcome, reranking status, and safe error type/code.

## Verification and troubleshooting

Use a real MCP client to verify initialization, schemas, and all tools. The
repository's deterministic profile suite exercises both transports:

```bash
uv run pytest tests/workflow_tests/test_oracle_knowledge_deployment_profiles.py -q
docker compose -f docker-compose.oracle-knowledge.yml config --quiet
```

If readiness is unavailable, check OCI config/profile, wallet/DSN, embedding
model configuration, and the container's stderr logs. `/health/live` should
remain available even when provider dependencies are unavailable.

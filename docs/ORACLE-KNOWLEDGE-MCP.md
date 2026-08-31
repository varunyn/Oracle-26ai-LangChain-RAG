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

The server reads the root `.env` through `api.settings`. Configure the OCI
embedding client, Oracle Vector connection, and friendly collection mapping:

```env
AUTH=API_KEY
OCI_PROFILE=DEFAULT
OCI_CONFIG_FILE=~/.oci/config
REGION=us-chicago-1
COMPARTMENT_ID=ocid1.compartment.oc1..replace-me
EMBED_MODEL_TYPE=OCI
EMBED_MODEL_ID=cohere.embed-v4.0

VECTOR_DB_USER=replace-me
VECTOR_DB_PWD=replace-me
VECTOR_DSN=replace-me
VECTOR_WALLET_DIR=/path/to/wallet
VECTOR_WALLET_PWD=replace-me

ORACLE_KNOWLEDGE_BASES={"default":"ORACLE_WEB_EMBEDDINGS"}
ORACLE_KNOWLEDGE_DEFAULT_KEY=default
ORACLE_KNOWLEDGE_TRANSPORT=stdio
ORACLE_KNOWLEDGE_HOST=127.0.0.1
ORACLE_KNOWLEDGE_PORT=9000
```

`SERVICE_ENDPOINT` is optional. When omitted, the app derives the OCI Generative
AI endpoint from `REGION`. `VECTOR_WALLET_DIR` and `VECTOR_WALLET_PWD` are needed
for wallet-based database connections. The OCI config must reference a valid API
key, or a container can supply `OCI_KEY_FILE` as an override.

The Oracle Knowledge settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ORACLE_KNOWLEDGE_BASES` | `{"default":"RAG_KNOWLEDGE_BASE"}` | Maps public friendly keys to Oracle collections. |
| `ORACLE_KNOWLEDGE_DEFAULT_KEY` | `default` | Selects the friendly key used when a tool call omits one. |
| `ORACLE_KNOWLEDGE_ALLOWED_KEYS` | unset | Optional comma-separated or JSON allowlist of public keys. |
| `ORACLE_KNOWLEDGE_CANDIDATE_LIMIT` | `20` | Default number of retrieval candidates. |
| `ORACLE_KNOWLEDGE_MAX_QUERY_LENGTH` | `8192` | Maximum query length. |
| `ORACLE_KNOWLEDGE_MAX_RESULT_LIMIT` | `50` | Maximum evidence results returned. |
| `ORACLE_KNOWLEDGE_MAX_CANDIDATE_LIMIT` | `100` | Maximum candidate limit accepted from callers. |
| `ORACLE_KNOWLEDGE_MAX_METADATA_FILTERS` | `8` | Maximum metadata filters accepted. |
| `ORACLE_KNOWLEDGE_TIMEOUT_SECONDS` | `30` | Tool execution timeout. |
| `ORACLE_KNOWLEDGE_ENABLE_RERANKER` | `true` | Enables OCI reranking after retrieval. |
| `ORACLE_KNOWLEDGE_ALLOW_RERANKER_OVERRIDE` | `false` | Allows callers to override reranking per request. |
| `ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING` | `false` | Enables tracing for the standalone process. |
| `ORACLE_KNOWLEDGE_TRANSPORT` | `stdio` | Selects `stdio` or `streamable-http`. |
| `ORACLE_KNOWLEDGE_HOST` | `127.0.0.1` | HTTP bind address. Ignored for STDIO. |
| `ORACLE_KNOWLEDGE_PORT` | `9000` | HTTP bind port. Ignored for STDIO. |

Only `stdio` and `streamable-http` are supported. SSE and other transports are
rejected. The standalone server uses only the namespaced transport settings.
Generic `TRANSPORT`, `HOST`, and `PORT` variables do not configure this server.

### Codex STDIO configuration

When Codex launches the server from the repository root, the process reads the
project `.env`; the Codex entry does not need to duplicate those values:

```toml
[mcp_servers.oracle_knowledge]
command = "uv"
args = ["run", "python", "/absolute/path/to/mcp_servers/oracle_knowledge.py"]
cwd = "/absolute/path/to/custom-rag-agent-app"
startup_timeout_sec = 20
tool_timeout_sec = 60
enabled = true
```

When `COMPARTMENT_ID` or an optional `SERVICE_ENDPOINT` comes from the Codex
process environment instead of the project `.env`, forward only those names:

```toml
env_vars = ["COMPARTMENT_ID", "SERVICE_ENDPOINT"]
```

If the launcher does not use the repository as its working directory, pass the
actual setting names from this page. `ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, and
`ORACLE_DSN` are not settings used by this server. Restart the MCP client after
changing its configuration or environment so it starts a new STDIO process.

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

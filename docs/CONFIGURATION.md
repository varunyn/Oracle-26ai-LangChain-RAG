# Configuration

Configure the RAG Agent API and frontend to match your environment and preferences.

This page is the **full reference**. If you are setting up the project for the first time, start with [GETTING-STARTED.md](GETTING-STARTED.md) and come back here when you need details on specific variables.

---

## Configuration sections

Jump to any section:

| Section                                                        | Description                                   |
| -------------------------------------------------------------- | --------------------------------------------- |
| [Configuration file locations](#configuration-file-locations)  | Where settings are read from and precedence   |
| [Quick start](#quick-start)                                    | Minimal steps to run backend and frontend     |
| [General / OCI](#general--oci)                                 | Auth and OCI GenAI environment                |
| [LLM & embeddings](#llm--embeddings)                           | Chat model, temperature, and embedding model  |
| [Oracle Vector Store](#oracle-vector-store)                    | Database connection for RAG search            |
| [RAG / Search](#rag--search)                                   | Retrieval and context options                 |
| [UI (backend)](#ui-backend)                                    | Language list and user feedback               |
| [Frontend (Next.js)](#frontend-nextjs)                         | Default flow mode, default model, backend URL |
| [Conversation memory / threads](#conversation-memory--threads) | Checkpointer and clear-chat behavior          |
| [MCP](#mcp)                                                    | MCP tools and server config                   |
| [Observability](#observability)                                | OpenTelemetry, Langfuse, Docker stacks        |
| [Usage in code](#usage-in-code)                                | How to read settings in the app               |
| [Complete example](#complete-example)                          | Sample `.env` excerpt                         |
| [Security notes](#security-notes)                              | Keeping secrets safe                          |

---

## Configuration file locations

The project reads settings from (in order of precedence):

1. **Environment variables** — Uppercase names (e.g. `REGION`, `LLM_MODEL_ID`).
2. **`.env` file** — In the project root. Copy from `.env.example` and set values.
3. **Built-in defaults** — Defined in `api/settings.py` when a value is not set.

**Precedence:** environment variables > `.env` > defaults.

The **frontend** also uses:

- **`frontend/.env.local`** — For `NEXT_PUBLIC_API_BASE`, `FASTAPI_BACKEND_URL` (optional fallback), and `NEXT_PUBLIC_DEFAULT_FLOW_MODE`. See `frontend/env.example`.

---

## Quick start

1. Copy the env template and set required values:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env`: set Oracle DB wallet + DSN, OCI profile, `COMPARTMENT_ID`, and any other options you need.
3. Run the backend:
   ```bash
   ./run_api.sh
   ```
4. Configure the frontend env (from `frontend/`):
   ```bash
   cp env.example .env.local
   ```
5. Run the frontend (from `frontend/`):
   ```bash
   pnpm install
   PORT=4000 pnpm dev
   ```

Recommended minimum variables for a first successful run:

- `VECTOR_DB_USER`
- `VECTOR_DB_PWD`
- `VECTOR_DSN`
- `VECTOR_WALLET_DIR`
- `VECTOR_WALLET_PWD`
- `OCI_PROFILE`
- `COMPARTMENT_ID`
- `REGION`
- `LLM_MODEL_ID`
- `EMBED_MODEL_ID`

---

## General / OCI

Control authentication and environment for OCI GenAI and DB access.

| Variable           | Default         | Description                                             |
| ------------------ | --------------- | ------------------------------------------------------- |
| `DEBUG`            | `false`         | Enable extra logging.                                   |
| `AUTH`             | `API_KEY`       | Auth mode: `API_KEY` or `SECURITY_TOKEN`.               |
| `OCI_PROFILE`      | `CHICAGO`       | Profile name in your OCI config.                        |
| `OCI_CONFIG_FILE`  | `~/.oci/config` | Path to OCI config file (supports repo-relative paths). |
| `REGION`           | `us-chicago-1`  | OCI region (affects model availability and endpoints).  |
| `COMPARTMENT_ID`   | (required)      | Your OCI compartment OCID (billing and access scope).   |
| `SERVICE_ENDPOINT` | (computed)      | Optional; computed from `REGION` if unset.              |

---

## LLM & embeddings

Choose the chat model for generation and the embedding model for search.

| Variable              | Default                       | Description                                                                               |
| --------------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `LLM_MODEL_ID`        | `meta.llama-3.3-70b-instruct` | Default chat model ID.                                                                    |
| `TEMPERATURE`         | `0.1`                         | Sampling temperature for responses.                                                       |
| `MAX_TOKENS`          | `4000`                        | Maximum tokens per response.                                                              |
| `MODEL_LIST`          | (by region)                   | Comma-separated or JSON array of model IDs. If unset, defaults are derived from `REGION`. |
| `MODEL_DISPLAY_NAMES` | `{}`                          | JSON object mapping model ID to display name (e.g. `{"model.id":"Display Name"}`).        |
| `EMBED_MODEL_TYPE`    | `OCI`                         | Embedding provider (only `OCI` supported).                                                |
| `EMBED_MODEL_ID`      | `cohere.embed-v4.0`           | Embedding model ID.                                                                       |

The frontend loads `MODEL_LIST` and `MODEL_DISPLAY_NAMES` from the backend via `GET /api/config` (no caching), so changes appear after restarting the backend and refreshing the page.

---

## Oracle Vector Store

Connect to Oracle 26AI for vector search (required for RAG).

| Variable                 | Default   | Description                                |
| ------------------------ | --------- | ------------------------------------------ |
| `VECTOR_DB_USER`         | —         | Database user.                             |
| `VECTOR_DB_PWD`          | —         | Database password.                         |
| `VECTOR_DSN`             | —         | Database DSN.                              |
| `VECTOR_WALLET_DIR`      | —         | Path to wallet directory.                  |
| `VECTOR_WALLET_PWD`      | —         | Wallet password.                           |
| `CONNECT_ARGS`           | (derived) | Optional; derived from the above if unset. |
| `DB_TCP_CONNECT_TIMEOUT` | `5`       | Connection timeout in seconds.             |

---

## RAG / Search

Control retrieval breadth and context shaping.

| Variable             | Default              | Description                                        |
| -------------------- | -------------------- | -------------------------------------------------- |
| `RAG_SEARCH_MODE`    | `vector`             | Retrieval mode for RAG: `vector`, `hybrid`, or `text`. |
| `COLLECTION_LIST`    | `RAG_KNOWLEDGE_BASE` | Comma-separated or JSON array of collection names. |
| `DEFAULT_COLLECTION` | `RAG_KNOWLEDGE_BASE` | Default collection when not specified.             |
| `CHUNK_SIZE`         | `4000`               | Chunk size for splitting.                          |
| `CHUNK_OVERLAP`      | `100`                | Overlap between chunks.                            |
| `ENABLE_RERANKER`    | `true`               | Enable reranking of retrieved chunks.              |

---

## UI (backend)

Backend options that affect the UI (e.g. dropdowns and features).

| Variable               | Default                                | Description                                        |
| ---------------------- | -------------------------------------- | -------------------------------------------------- |
| `ENABLE_USER_FEEDBACK` | `true`                                 | Enable in-UI feedback (e.g. star rating).          |

---

## Frontend (Next.js)

The frontend gets most config from the backend via `GET /api/config`. These options are **frontend-only** and live in `frontend/.env.local` (see `frontend/env.example`).

| Variable                        | Default                 | Description                                                          |
| ------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `NEXT_PUBLIC_API_BASE`          | `http://localhost:3002` | Browser-visible backend base URL for direct frontend API calls.      |
| `FASTAPI_BACKEND_URL`           | `http://localhost:3002` | Optional server-side fallback for config/bootstrap fetches.          |
| `NEXT_PUBLIC_DEFAULT_FLOW_MODE` | `rag`                   | Default flow when the app loads: `rag`, `mcp`, `mixed`, or `direct`. |

### Default model (browser)

The model the user selects in the UI is stored in **browser localStorage** under the key `rag_default_model` and is restored on refresh and across server restarts. There is no env var: the “default model” is whatever the user last chose. If the stored model is no longer in the backend’s `MODEL_LIST`, it is replaced with the first model from the list.

---

## Conversation memory / threads

Thread state behavior.

| Variable                   | Default          | Description                                                       |
| -------------------------- | ---------------- | ----------------------------------------------------------------- |
| `ALLOW_CLIENT_THREAD_ID`   | (when supported) | Allow client to send thread ID.                                   |
| `THREAD_ID_STRATEGY`       | (when supported) | How thread IDs are generated.                                     |
| `THREAD_ID_PREFIX`         | (when supported) | Optional prefix for thread IDs.                                   |

### Clear chat

When the user clears the chat, the frontend calls **`DELETE /api/threads/{thread_id}`** to remove the thread’s server-side runtime state, then clears local UI state and starts a new thread. A success toast is shown.

---

## MCP

MCP (Model Context Protocol) client and server configuration.

| Variable                            | Default        | Description                                                    |
| ----------------------------------- | -------------- | -------------------------------------------------------------- |
| `ENABLE_MCP_TOOLS`                  | `true`         | Enable MCP tool use.                                           |
| `MCP_SERVER_KEYS`                   | (none)         | Comma-separated list of configured MCP server keys to load tools from (e.g. `default,context7`). This does not choose the default chat mode. |
| `MCP_TOOL_SELECTION_MAX_TOOLS`      | `5`            | Max tools to select per turn.                                  |
| `MCP_TOOL_SELECTION_ALWAYS_INCLUDE` | `[]`           | Tool names always included (JSON array).                       |
| `MCP_SEARCH_MODE`                   | `vector`       | Default retrieval mode for semantic-search MCP tools: `vector`, `hybrid`, or `text`. |
| `ENABLE_MCP_CLIENT_JWT`             | `false`        | Enable JWT auth for MCP client.                                |
| `MCP_SERVERS_CONFIG`                | (see settings) | JSON object; see [MCP-USAGE.md](MCP-USAGE.md).                 |

### MCP server runtime (`mcp_servers/*.py`)

| Variable    | Default           | Description                              |
| ----------- | ----------------- | ---------------------------------------- |
| `TRANSPORT` | `streamable-http` | Transport: `streamable-http` or `stdio`. |
| `HOST`      | `0.0.0.0`         | Bind host for MCP server.                |
| `PORT`      | `9000`            | Bind port for MCP server.                |

---

## Observability

Optional tracing, logging, and local Docker stacks.

| Variable                       | Default                 | Description                                             |
| ------------------------------ | ----------------------- | ------------------------------------------------------- |
| `ENABLE_OTEL_TRACING`          | `false`                 | Enable OpenTelemetry tracing.                           |
| `OTEL_TRACES_ENDPOINT`         | —                       | OTLP traces endpoint.                                   |
| `OTEL_TRACES_HEADERS`          | —                       | Headers for traces (e.g. auth).                         |
| `OTEL_LOGS_ENDPOINT`           | —                       | OTLP logs endpoint.                                     |
| `ENABLE_OBSERVABILITY_STACK`   | `false`                 | Start local observability stack (Grafana, Tempo, etc.). |
| `ENABLE_LANGFUSE_TRACING`      | `false`                 | Send traces to Langfuse.                                |
| `LANGFUSE_HOST`                | `http://localhost:3300` | Langfuse server URL.                                    |
| `LANGFUSE_PUBLIC_KEY`          | —                       | Langfuse public key.                                    |
| `LANGFUSE_SECRET_KEY`          | —                       | Langfuse secret key.                                    |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development`           | Langfuse tracing environment name.                      |

**Langfuse stack (Docker)**

To run the Langfuse stack with Docker (e.g. `make langfuse-up`), you **must** set up a separate `.env` for the stack:

- Create `observability/langfuse/.env` (copy from `observability/langfuse/.env.example`).
- Set all required values (e.g. database URLs, secrets, `NEXTAUTH_URL`).
- Without this file, the Langfuse containers will fail to start or run with invalid defaults.

**Two different env files:**

| Location                      | Used by                 | Purpose                                                                                       |
| ----------------------------- | ----------------------- | --------------------------------------------------------------------------------------------- |
| Project root `.env`           | Backend (RAG API)       | `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` for sending traces to Langfuse. |
| `observability/langfuse/.env` | Langfuse stack (Docker) | Database, auth, and service config for the Langfuse containers.                               |

See [OBSERVABILITY.md](OBSERVABILITY.md) and [OBSERVABILITY_ROUTING.md](OBSERVABILITY_ROUTING.md).

### Docker stacks (optional)

| Variable        | Default        | Description                                                         |
| --------------- | -------------- | ------------------------------------------------------------------- |
| `DOCKER_STACKS` | (see settings) | JSON object; which stacks to manage. Override via `.env` if needed. |

- Use **Makefile:** `make up`, `make down`, `make status` (or `make observability-up`, `make langfuse-up`, etc.). See [DOCKER-SETUP.md](DOCKER-SETUP.md).
- Or run **scripts:** `uv run python scripts/manage_stacks.py up | down | status`.
- **Auto-included stacks** when `DOCKER_STACKS` is not set: `observability` if `ENABLE_OBSERVABILITY_STACK=true` or `ENABLE_OTEL_TRACING=true`; `langfuse` if `ENABLE_LANGFUSE_TRACING=true`.

### OCI Logging Analytics (optional)

| Variable                         | Default | Description                         |
| -------------------------------- | ------- | ----------------------------------- |
| `ENABLE_OCI_LOGGING_ANALYTICS`   | `false` | Send logs to OCI Logging Analytics. |
| `LOGGING_ANALYTICS_NAMESPACE`    | —       | Logging Analytics namespace.        |
| `LOGGING_ANALYTICS_LOG_GROUP_ID` | —       | Log group OCID.                     |
| (others)                         | —       | See `.env.example` for full list.   |

---

## Usage in code

- **Backend:** Read settings via `from api.settings import get_settings`. All configuration is type-checked and validated at startup.
- **Frontend:** Config is fetched from `GET /api/config` and provided via the config provider; see `frontend/src/lib/config.ts` and `frontend/src/components/config-provider.tsx`.

---

## Complete example

Minimal `.env` excerpt for local development:

```env
# General / OCI
DEBUG=false
AUTH=API_KEY
OCI_PROFILE=CHICAGO
REGION=us-chicago-1
COMPARTMENT_ID=ocid1.compartment.oc1..your-compartment-ocid

# LLM
LLM_MODEL_ID=meta.llama-3.3-70b-instruct
TEMPERATURE=0.1
MAX_TOKENS=4000

# Embeddings
EMBED_MODEL_TYPE=OCI
EMBED_MODEL_ID=cohere.embed-v4.0

# Oracle Vector Store (required for RAG)
VECTOR_DB_USER=your_user
VECTOR_DB_PWD=your_password
VECTOR_DSN=your_dsn
VECTOR_WALLET_DIR=/path/to/wallet
VECTOR_WALLET_PWD=wallet_password
DB_TCP_CONNECT_TIMEOUT=5

# RAG
RAG_SEARCH_MODE=vector
COLLECTION_LIST=RAG_KNOWLEDGE_BASE
ENABLE_RERANKER=true

# MCP (semantic-search tools)
MCP_SEARCH_MODE=vector
```

See `.env.example` in the project root for all supported variables and comments.

For the shortest reproducible local path, pair this page with [GETTING-STARTED.md](GETTING-STARTED.md).

---

## Security notes

- **Never commit `.env` or secrets.** `.env` is gitignored.
- Use `local-config/` for OCI config and wallet when running locally; mount secrets in Docker for production.
- Prefer environment variables in CI/CD over committed files.

---

## Related docs

| Doc                                                  | Description                                                |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| [MCP-USAGE.md](MCP-USAGE.md)                         | MCP servers and tool usage                                 |
| [OBSERVABILITY.md](OBSERVABILITY.md)                 | Tracing and observability setup                            |
| [OBSERVABILITY_ROUTING.md](OBSERVABILITY_ROUTING.md) | Routing traces/logs to Grafana, OCI APM, Logging Analytics |
| [AGENTS.md](../AGENTS.md)                            | Commands, lint, and project overview                       |

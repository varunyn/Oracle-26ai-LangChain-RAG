# Getting started (first run)

This guide walks a **first-time user** from zero to a working chat session. It assumes you are running locally.

> This project is a production-style reference app, not a zero-config starter. Expect to provide real Oracle DB, wallet, and OCI access before the chat flow works end to end.

## Prerequisites

- **Python 3.11**
- **uv** (Python package manager)
- **pnpm** (frontend package manager)
- **Oracle Database 23AI/26AI** with vector support + wallet
- **OCI account** with Generative AI access
- **Docker** (optional; only for local observability or Langfuse)

If you do not have an Oracle DB yet, start with [DATABASE-SETUP.md](DATABASE-SETUP.md).

## Recommended first-run path

For the smoothest first run, use this order:

1. Configure backend `.env`
2. Create the vector table (if needed)
3. Ingest a small set of documents
4. Start the backend with `./run_api.sh`
5. Start the frontend on port `4000`

If you prefer Docker-managed services, see [DOCKER-SETUP.md](DOCKER-SETUP.md) after finishing the basic local path once.

## 1. Install backend dependencies

From the repo root:

```bash
uv sync
```

Optional dev tools:

```bash
uv sync --group dev
```

## 2. Configure backend environment

Copy the template and set required values:

```bash
cp .env.example .env
```

At minimum, set these in `.env`:

- **Database**: `VECTOR_DB_USER`, `VECTOR_DB_PWD`, `VECTOR_DSN`, `VECTOR_WALLET_DIR`, `VECTOR_WALLET_PWD`
- **OCI**: `OCI_PROFILE`, `COMPARTMENT_ID`, `REGION`
- **Models**: `LLM_MODEL_ID`, `EMBED_MODEL_ID` (defaults exist; change if needed)

See [CONFIGURATION.md](CONFIGURATION.md) for the full matrix.

## 3. Configure frontend environment

From the `frontend/` directory:

```bash
cp env.example .env.local
```

Set `FASTAPI_BACKEND_URL` if your API is not on `http://localhost:3002`.

You usually do **not** need any other frontend env values for a basic local run.

## 4. Create the vector table (once)

If you do **not** already have a vector table, create one with the built-in script:

```bash
uv run scripts/create_rag_table.py --table RAG_KNOWLEDGE_BASE --yes
```

If you need a specific embedding dimension, pass `--embedding-dim` (see the script header for examples).

## 5. Load documents into the vector store

Place documents (PDF, HTML, TXT, MD) in a local folder, then ingest. The shared ingestion logic lives in `src/rag_agent/ingestion.py`, and the script below is the supported CLI wrapper for local batch loads:

```bash
uv run python scripts/ingest_documents.py --dir ./documents
```

Or pass specific files:

```bash
uv run python scripts/ingest_documents.py --files doc1.pdf notes.txt readme.md
```

## 6. Run the backend API

```bash
./run_api.sh
```

The API defaults to **http://localhost:3002**.

If the API fails to start, stop here and fix backend/database/OCI configuration before starting the frontend. A running frontend without a working backend is not a meaningful validation of the app.

## 7. Run the frontend

```bash
cd frontend
pnpm install
PORT=4000 pnpm dev
```

Open **http://localhost:4000** and ask a question.

For the first successful end-to-end check, ask a question that should clearly match one of the documents you ingested.

## Optional: MCP tools (advanced)

If you want MCP tools available in the chat, start an MCP server in a separate terminal:

```bash
uv run python mcp_servers/mcp_semantic_search.py
```

Then ensure `ENABLE_MCP_TOOLS=true` and `MCP_SERVERS_CONFIG` are set in `.env` (see [MCP-USAGE.md](MCP-USAGE.md)). For a local standalone MCP server from this repo, the canonical default URL is `http://localhost:9000/mcp`.

## Troubleshooting

- **No answers / empty search results** → ensure documents were ingested and the collection exists.
- **Database errors** → verify wallet path and `VECTOR_*` values in `.env`.
- **Frontend can’t reach API** → set `FASTAPI_BACKEND_URL` in `frontend/.env.local`.
- **The app runs but answers are weak or empty** → confirm the collection name, ingestion step, and `RAG_SEARCH_MODE` / `TOP_K` settings.

For deeper setup details, see:

- [CONFIGURATION.md](CONFIGURATION.md)
- [DATABASE-SETUP.md](DATABASE-SETUP.md)
- [DOCUMENT-POPULATION.md](DOCUMENT-POPULATION.md)

# Custom RAG Agent

A production-ready **Retrieval-Augmented Generation (RAG)** agent built with **LangGraph**, **Oracle 26AI Vector Store**, and **OCI Generative AI**. This agent implements an advanced multi-step workflow for intelligent document Q&A with streaming responses.

## Overview

This application provides an intelligent question-answering system that:

- Processes user queries through a multi-stage pipeline
- Searches documents using semantic vector search
- Reranks results using LLM-based relevance scoring
- Generates contextual answers with citations
- Supports streaming responses and real-time UI updates

## Architecture

Router branches by `route`: **search** (handled through `FollowUpInterpreter`, which then either rewrites for retrieval or reformats from grounded context), **select_mcp** (MCP tools path), or **direct** (LLM-only). In `mixed` mode, the workflow can start with the retrieval path and fall back to MCP when the RAG answer is weak or missing citations.

### Key Directories

| Directory        | Purpose                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `src/rag_agent/` | LangGraph workflow, follow-up interpretation, search, reranker, answer generator, router, MCP nodes |
| `api/`           | FastAPI app, chat/config/documents/feedback/health routers, graph invocation                  |
| `frontend/`      | Next.js app; `src/app` (pages, API routes), `src/components`, `src/lib` (chat, config, types) |
| `mcp_servers/`   | MCP servers (RAG, semantic search, minimal)                                                   |
| `scripts/`       | Document population, table create/drop/truncate, BM25                                         |
| `tests/`         | Pytest and manual run scripts for MCP/workflow                                                |
| `docs/`          | Setup, MCP usage, tracing, OCI, database                                                      |

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    Router(Router)
    Search(Search)
    SearchErrorResponse(SearchErrorResponse)
    Rerank(Rerank)
    AnswerFromDocs(AnswerFromDocs)
    DraftAnswer(DraftAnswer)
    FollowUpInterpreter(FollowUpInterpreter)
    GroundedReformatAnswer(GroundedReformatAnswer)
    SelectMCPTools(SelectMCPTools)
    CallMCPTools(CallMCPTools)
    DirectAnswer(DirectAnswer)
    __end__([<p>__end__</p>]):::last
    AnswerFromDocs -. &nbsp;draft&nbsp; .-> DraftAnswer;
    AnswerFromDocs -. &nbsp;select_mcp&nbsp; .-> SelectMCPTools;
    CallMCPTools --> DraftAnswer;
    DirectAnswer --> DraftAnswer;
    FollowUpInterpreter -. &nbsp;reformat&nbsp; .-> GroundedReformatAnswer;
    FollowUpInterpreter -. &nbsp;search&nbsp; .-> Search;
    GroundedReformatAnswer --> DraftAnswer;
    Rerank --> AnswerFromDocs;
    Router -. &nbsp;direct&nbsp; .-> DirectAnswer;
    Router -. &nbsp;followup&nbsp; .-> FollowUpInterpreter;
    Router -. &nbsp;select_mcp&nbsp; .-> SelectMCPTools;
    Search -. &nbsp;error&nbsp; .-> SearchErrorResponse;
    Search -. &nbsp;rerank&nbsp; .-> Rerank;
    SelectMCPTools --> CallMCPTools;
    __start__ --> Router;
    DraftAnswer --> __end__;
    SearchErrorResponse --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Key Components

### 1. **Semantic Search** (`vector_search.py`)

- Performs retrieval in Oracle 23AI with configurable mode
- Supports `vector`, `hybrid` (vector + lexical fusion), or `text` retrieval
- Uses embedding model for vector component and returns top K relevant chunks

### 2. **Reranker** (`reranker.py`)

- Uses LLM to evaluate and rank retrieved documents
- Filters out irrelevant results
- Improves answer quality by focusing on best matches

### 3. **Answer Generator** (`answer_generator.py`)

- Generates final answer using retrieved context
- Includes citations to source documents
- Supports streaming for real-time responses

### 5. **State Management** (`agent_state.py`)

- Manages workflow state across all nodes
- Tracks: user request, chat history, documents, answers, errors

## Data Flow

```
User Query
    ↓
[Query Rewriter] → Reformulates using history
    ↓
[Vector Search] → Finds similar documents
    ↓
[Reranker] → Filters & ranks documents
    ↓
[Answer Generator] → Creates final answer
    ↓
Next.js UI → Displays answer + citations
```

## Technology Stack

- **Framework**: LangGraph (agent orchestration)
- **Vector Database**: Oracle 26AI with VECTOR data type
- **LLM**: OCI Generative AI (Meta Llama, Cohere, OpenAI models)
- **Embeddings**: OCI Generative AI (Cohere multilingual)
- **UI**: Next.js
- **Observability**: OpenTelemetry (OTLP); OCI APM supported via OTLP
- **Language**: Python 3.11

## Setup

**New here?** Start with the step-by-step guide: [GETTING-STARTED.md](GETTING-STARTED.md).

### Prerequisites

1. **Oracle 26AI Database** with:
   - Vector Store enabled
   - Table: `RAG_KNOWLEDGE_BASE` (created automatically)
   - Wallet configured for secure connection

2. **OCI Account** with:
   - Generative AI service access
   - API keys configured in `~/.oci/config`
   - Compartment with Generative AI permissions

3. **Python 3.11**
4. **uv** (Python package manager)
5. **pnpm** (frontend package manager)

### Installation

This project uses `uv` for package management with `pyproject.toml` as the source of truth.

```bash
# Install uv (if not already installed)
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# Sync project dependencies (creates .venv, installs all dependencies, generates uv.lock)
uv sync
```

**Note**: The project uses `uv` and `pyproject.toml` for dependency management. Use `uv run` to run commands so the project virtualenv is used automatically.

OCI and Oracle AI Vector Search integrations use the official [oracle/langchain-oracle](https://github.com/oracle/langchain-oracle) packages: **langchain-oci** (LLM and embeddings) and **langchain-oracledb** (vector store). See that repository for documentation and examples.

**OCI Gen AI** is used via **ChatOCIGenAI** (from langchain-oci) for RAG (answer, reranker, follow-up interpretation) and MCP tool-calling. Auth uses the OCI profile from config (~/.oci/config).

**Development dependencies**:

```bash
# Install with development tools (pytest, black, ruff, mypy)
uv sync --group dev
```

### Configuration

**IMPORTANT**: Copy `.env.example` to `.env` and set your values. The `.env` file is in `.gitignore` and will not be committed.

1. **Create your env file**:

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** – set at least:
   - **Database**: `VECTOR_DB_USER`, `VECTOR_DB_PWD`, `VECTOR_DSN`, `VECTOR_WALLET_DIR`, `VECTOR_WALLET_PWD`
   - **OCI**: `OCI_PROFILE`, `COMPARTMENT_ID`, `REGION`
   - **Models**: `LLM_MODEL_ID`, `EMBED_MODEL_ID` (defaults exist; override as needed)

   See [CONFIGURATION](CONFIGURATION.md) and `.env.example` for all options.

## Usage

### 1. Populate Knowledge Base

The ingestion implementation lives in `src/rag_agent/ingestion.py`. For local operations and batch ingestion, the supported CLI entrypoint remains `scripts/ingest_documents.py`, which now wraps that shared module.

```bash
# Process specific files (PDF, HTML, TXT, MD)
uv run python scripts/ingest_documents.py --files document1.pdf document2.pdf readme.md

# Process all supported files in a directory
uv run python scripts/ingest_documents.py --dir ./documents
```

### 2. Run the Application

#### Local ports

| Service            | URL                   | Notes                                    |
| ------------------ | --------------------- | ---------------------------------------- |
| Backend (FastAPI)  | http://localhost:3002 | Default API port                         |
| Frontend (Next.js) | http://localhost:4000 | Repo standard for dev and Docker         |
| Grafana            | http://localhost:3051 | Only when observability stack is enabled |
| Langfuse UI        | http://localhost:3300 | Only when Langfuse stack is enabled      |

#### Option A – Local processes

```bash
# Terminal 1 – FastAPI backend
./run_api.sh

# Terminal 2 – Next.js UI (port 4000)
cd frontend
pnpm install
cp env.example .env.local
PORT=4000 pnpm dev
```

#### Option B – Docker Compose (backend + frontend)

```bash
docker compose up -d backend frontend
# or just `docker compose up -d` to include any other services defined
```

- API: http://localhost:3002 (default; override with `PORT` env var)
- Frontend: http://localhost:4000 (container exposes port 3000 at 4000 per compose)
- Logs: `docker compose logs -f backend` (or `frontend`)
- Stop: `docker compose down`

#### Optional observability stack (Grafana/Loki/Tempo)

```bash
docker compose --profile observability up -d loki tempo otel-collector grafana
# stop:
docker compose --profile observability down
```

This starts the collector + Loki + Tempo + Grafana defined in `docker-compose.yml`.

If you are running the API locally (Option A), you can also have `./run_api.sh` start these containers by setting `ENABLE_OBSERVABILITY_STACK=true` in `.env`.

#### Optional Langfuse stack

If you want Langfuse SDK traces locally:

```bash
cp observability/langfuse/.env.example observability/langfuse/.env
# edit secrets
docker compose -f observability/langfuse/docker-compose.yml up -d
```

The Langfuse UI will run at `http://localhost:3300` (default) using its own compose file so it doesn't interfere with the main stack. See `observability/langfuse/README.md` for details.

- API: http://localhost:3002 (default; override with `PORT` env var)
- Frontend: http://localhost:4000 (Next.js dev server reads API URL from env)

#### Optional one-command stack management

1. Stacks are defined in `api/settings.py` (DOCKER_STACKS). Override in `.env` if needed (JSON).
2. Use the helper script:

   ```bash
   # Bring up every stack with enabled=True
   uv run python scripts/manage_stacks.py up

   # Target specific stacks
   uv run python scripts/manage_stacks.py up --stacks core
   uv run python scripts/manage_stacks.py status --stacks langfuse
   uv run python scripts/manage_stacks.py down --stacks observability langfuse
   ```

3. The script uses `get_settings().DOCKER_STACKS` and shells out to `docker compose`.
   If no stacks are specified, it also auto-includes `observability` when
   `ENABLE_OBSERVABILITY_STACK=true` or `ENABLE_OTEL_TRACING=true`, and
   `langfuse` when `ENABLE_LANGFUSE_TRACING=true`.

### 3. Query the Knowledge Base

1. Enter your question in the chat interface
2. The agent processes through all workflow stages
3. View intermediate results in the sidebar:
   - Standalone question (when a follow-up is rewritten for retrieval)
   - References (after reranking)
4. Receive streaming answer with citations

## Features

### ✨ Streaming Responses

- Real-time answer generation
- Progressive UI updates as each stage completes
- Immediate display of document references

### 🔄 Chat History and Memory

- Maintains conversation context (follow-up interpreter and answer generator use `chat_history` from state)
- Rewrites retrieval-oriented follow-up questions into standalone questions when needed
- Configurable history length (`MAX_MSGS_IN_HISTORY`)
- **LangGraph checkpointer**: state (including chat history) is persisted per `thread_id` via SQLite/AsyncSqliteSaver in the API runtime, and persisted message history is capped by `MAX_MSGS_IN_HISTORY` to avoid unbounded growth

### 🎯 Intelligent Reranking

- LLM-based relevance scoring
- Filters out irrelevant documents
- Improves answer accuracy

### 📊 Observability (Optional)

- OCI APM integration for tracing
- Performance monitoring
- Error tracking

### 🔒 Security

- Wallet-based database authentication
- OCI profile for GenAI; no secrets in repo (use `.env` from `.env.example`)

**OCI keys (Docker best practice: use [Secrets](https://docs.docker.com/compose/use-secrets/) for API keys, not env vars for key content):**

- **Without Docker:** Keys in local files. Use `local-config/oci/config` with `key_file=../oci_api_key.pem` (relative to config file) so the same config works locally and in Docker.
- **With Docker:** Compose uses a secret for the key (mounted at `/run/secrets/oci_api_key`); the app is given the path via `OCI_KEY_FILE`. Key content is never in the image or in environment variable values. Config and wallet remain in the `./local-config` volume.

## MCP (Model Context Protocol) Integration

The application includes **MCP server** support, allowing LLM agents to interact with the vector database through standardized tools. This enables external agents (like Claude Desktop, custom LLM applications) to perform semantic search and query your knowledge base.

### MCP User Flow

```mermaid
sequenceDiagram
    participant User
    participant LLM as LLM Agent
    participant Client as MCP Client
    participant Server as MCP Server
    participant DB as Oracle Vector DB
    participant Embed as Embedding Model

    User->>LLM: Ask Question
    LLM->>Client: Discover Available Tools
    Client->>Server: list_tools()
    Server-->>Client: Tool Schemas

    LLM->>Client: Call semantic_search(query, search_mode)
    Client->>Server: POST /mcp/

    Server->>Embed: Generate Embeddings
    Embed-->>Server: Query Vector
    Server->>DB: Vector Similarity Search
    DB-->>Server: Top K Documents
    Server-->>Client: JSON Response
    Client-->>LLM: Tool Results
    LLM->>LLM: Generate Answer
    LLM-->>User: Final Answer with Context
```

### MCP Tools

The MCP server exposes three main tools:

1. **`semantic_search`** - Search for relevant documents
   - Parameters: `query`, `top_k`, `collection_name` (optional), `search_mode` (optional: `vector`/`hybrid`/`text`)
   - Returns: Relevant document chunks with metadata

2. **`get_collections`** - List available collections
   - Returns: List of vector table names in the database

3. **`list_documents_in_collection`** - List documents in a collection
   - Parameters: `collection_name` (optional)
   - Returns: List of unique document sources with chunk counts

The **RAG MCP server** (`mcp_rag_server.py`) exposes **`rag_ask`** for full RAG (query → search → rerank → answer with citations).

### Using MCP

See [MCP usage](MCP-USAGE.md) for usage guide.

## Advantages of Agentic Approach

The modular LangGraph architecture provides:

1. **Flexibility**: Easy to add/remove/modify workflow steps
2. **Observability**: Each step can be monitored independently
3. **Error Handling**: Graceful degradation at each stage
4. **Extensibility**: Simple to add features like:
   - PII detection and anonymization
   - Multi-language support
   - Custom filtering logic
   - Additional validation steps

## Example Workflow Execution

```
User: "What is Oracle 23AI?"

1. [Search] → Found 6 relevant document chunks
2. [Rerank] → Ranked and filtered to top 3 chunks
3. [Answer] → Generated answer with citations:

   "Oracle 23AI is Oracle's next-generation database..."

   References:
   - Oracle Docs (page 5)
```

## Documentation

- [Getting started](GETTING-STARTED.md) – First run walkthrough
- [Docker setup](DOCKER-SETUP.md) – Run services with Docker/compose
- [Database setup](DATABASE-SETUP.md) – Vector DB and wallet configuration
- [Document population](DOCUMENT-POPULATION.md) – Ingesting documents into the knowledge base
- [MCP usage](MCP-USAGE.md) – Using MCP tools and RAG MCP server
- [OCI session token](OCI-SESSION-TOKEN.md) – OCI session token auth
- [Tracing](TRACING.md) – Observability and tracing
- [Observability routing](OBSERVABILITY_ROUTING.md) – How to combine local Grafana/Tempo, OCI APM, and OCI Logging Analytics

### Documentation site (GitHub Pages)

This docs folder is a **Docsify** site. To publish on GitHub Pages:

1. In your repo: **Settings → Pages → Build and deployment**
2. Under **Source**, choose **Deploy from a branch**
3. Branch: **main** (or default), Folder: **/docs**
4. Save. The site will be at `https://<username>.github.io/<repo-name>/`

**View locally:** From the repo root, run `./scripts/serve_docs.sh` (or `python -m http.server 3333 --directory docs`), then open **http://localhost:3333** in your browser.

## Troubleshooting

See the [Documentation](#documentation) section above. For database or OCI issues, start with [DATABASE-SETUP](DATABASE-SETUP.md) and [OCI-SESSION-TOKEN](OCI-SESSION-TOKEN.md).

## Contributing

See [AGENTS.md](AGENTS.md) (in the repo root) for contribution workflow, testing gate, and code style.

## License

MIT License

## References

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Oracle LangChain integration (langchain-oci, langchain-oracledb)](https://github.com/oracle/langchain-oracle)
- [Oracle 23AI Vector Search](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)
- [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)

# Oracle RAG Agent Blog Design

## Purpose

Create a technical blog draft for Oracle/OCI developers who want to understand how this application combines FastAPI, Oracle AI Vector Search, OCI Generative AI, MCP tools, and a Next.js chat UI.

The article should be architecture-first and app-focused. It should explain what the app does, how its major runtime pieces fit together, and why those product and architecture decisions matter. Setup commands should be avoided in the article body; link to the existing setup docs instead of duplicating operational steps.

## Deliverable

- File: `docs/blog/oracle-rag-agent-fastapi-oci-genai.md`
- Format: Markdown article stored in this repo
- Image handling: use local screenshots from `images/` first, then add fresh screenshots only where the current assets do not cover a key UI surface

## Audience

Primary audience: Oracle/OCI developers building RAG or agentic applications with:

- Oracle Database 23ai or 26ai vector search
- Oracle Autonomous Database style wallet-based connectivity
- OCI Generative AI chat, embeddings, and reranking
- FastAPI backend services
- Optional MCP tool integrations

The reader should come away understanding the implementation architecture and where to look in the repo to run or adapt it.

## Recommended Article Shape

Working title:

`Building an Oracle RAG Agent with FastAPI, Oracle AI Vector Search, OCI Generative AI, and MCP`

Sections:

1. Why a normal chat app is not enough for enterprise RAG
2. The architecture: Next.js UI, FastAPI API, `ChatRuntimeService`, Oracle vector search, OCI Generative AI, MCP, and observability
3. Preparing Oracle AI Vector Search: table shape, wallet config, collection settings, and ingestion
4. FastAPI as the reusable backend layer: config, documents, chat, feedback, health, and LangGraph-compatible streaming
5. Runtime modes: `rag`, `mcp`, `mixed`, and `direct`
6. Mixed mode: combining Oracle retrieval and MCP tools in one agent loop
7. The UI surfaces: model selection, flow mode, document upload, processed sources, and settings
8. Observability: request IDs, traces, Langfuse, OpenTelemetry, and why agent runs need more than HTTP logs
9. Trade-offs and limitations
10. How to try it locally, with links to the repo docs

## Screenshot Plan

Primary visual:

- `images/oracle-rag-agent-hand-drawn-architecture.png`

Use the hand-drawn architecture image as the main system explainer because it is more distinctive and draws more attention than a plain flow diagram. Use Mermaid only as supporting material when a precise sequence or runtime branch is easier to read as a simple flow.

Use existing screenshots:

- `images/oci-custom-rag-agent-chat-upload-panel.png`
- `images/oci-custom-rag-agent-processed-sources-table.png`
- `images/oci-custom-rag-agent-flow-mode-selector.png`
- `images/oci-custom-rag-agent-model-selector.png`

Take fresh screenshots if the app can be run locally:

- Settings page for UI-managed MCP server configuration
- Optional Langfuse trace view, only if the local stack is running and contains safe demo data

Do not include screenshots with credentials, tokens, private document contents, or personal data.

## Technical Grounding

Ground implementation claims in the current repo:

- `src/rag_agent/runtime/chat_service.py` owns chat runtime dispatch and thread state behavior.
- `api/` exposes the FastAPI application, config, documents, feedback, health, and LangGraph-compatible endpoints.
- `src/rag_agent/infrastructure/oci_models.py` and related infrastructure modules handle OCI model and reranker wiring.
- Oracle vector retrieval uses `langchain-oracledb` OracleVS-compatible tables.
- MCP configuration is primarily UI-managed, with `.env` used as optional seed or headless config.
- Streaming behavior must be described as a frontend/backend contract, not an incidental UI detail.

## Boundaries

Include:

- The hand-drawn architecture image as the lead diagram
- Plain Markdown/Mermaid flow diagrams only where they clarify a specific runtime branch
- Links to operational docs instead of inline backend/frontend setup commands
- Links to `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`, `docs/DATABASE-SETUP.md`, `docs/DOCUMENT-POPULATION.md`, `docs/MCP-USAGE.md`, and `docs/OBSERVABILITY.md`
- Honest trade-offs around setup complexity, OCI dependencies, MCP configuration, and streaming contracts

Avoid:

- Copying the reference article's wording
- Turning the post into a complete setup manual
- Claiming production readiness beyond what the repo demonstrates
- Publishing secrets, wallet paths, real trace payloads, or private document data

## Success Criteria

- The draft is useful to an OCI developer evaluating or adapting the app.
- The article explains the real architecture rather than a generic RAG stack.
- Screenshots are locally referenced and safe to publish.
- The post links to deeper docs instead of duplicating them.
- The tone is direct, technical, and specific.

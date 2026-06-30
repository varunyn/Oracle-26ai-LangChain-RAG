# MCP Servers AGENTS.md - Local Codex Hints

This file applies only to `mcp_servers/`. Root `AGENTS.md` owns repo-wide rules.

## Server Map

These files expose standalone FastMCP servers from this repo:

- `mcp_semantic_search.py`: semantic search, collection listing, and collection document listing.
- `mcp_rag_server.py`: RAG workflow exposed as the `rag_ask` tool.
- `run_mcp_minimal.py` is a manual smoke script under `tests/`, not a server implementation.

Do not confuse these standalone exposed servers with the app's MCP-consuming chat path. App chat consumes configured MCP servers through the LangGraph Agent Server `chat_agent` graph on port 2024, with `mode="mcp"` or `mode="mixed"` passed in run context.

## Where To Look

- `src/rag_agent/infrastructure/mcp_settings.py`: MCP transport normalization and client/server config conventions.
- `src/rag_agent/infrastructure/mcp_adapter_runtime.py`: app-side MCP client wiring through `langchain_mcp_adapters`.
- `docs/MCP-USAGE.md`: expose-vs-consume behavior and local usage examples.
- `docs/CONFIGURATION.md`: MCP env vars, including server runtime settings.

## Runtime Config

- Standalone servers read settings through `api.settings.get_settings()`.
- `TRANSPORT` supports `streamable-http` and `stdio`; normalize with `normalize_mcp_transport(...)`.
- HTTP servers use `HOST`, `PORT`, and the `/mcp` path. Local default is typically `http://localhost:9000/mcp`.
- Consuming-side server selection belongs in the Settings UI config store (with env/config fallback for headless operation). Do not hardcode consumed server URLs in standalone server code.
- Never commit credentials, wallet paths, tokens, or environment-specific external endpoints.

## Run And Smoke Test

Semantic search server:

```bash
uv run python mcp_servers/mcp_semantic_search.py
uv run python tests/run_mcp_semantic_search.py
```

RAG server:

```bash
uv run python mcp_servers/mcp_rag_server.py
uv run python tests/run_mcp_rag.py
```

The FastAPI app does not need to be running to start these standalone servers, but the underlying retrieval/RAG dependencies still need valid local config. Check `docker compose ps` first and reuse any already-running database/backend support services when relevant.

## Boundaries

- Keep tools small, typed, and JSON-serializable.
- Keep standalone MCP tool implementations in `mcp_servers/*`; do not hide them in FastAPI handlers.
- Keep app-side MCP consumption in `src/rag_agent/infrastructure/mcp_adapter_runtime.py` and runtime orchestration layers.
- Update `docs/MCP-USAGE.md`, `docs/CONFIGURATION.md`, and matching `tests/run_mcp_*.py` scripts when tool names, server files, transports, or payload shapes change.
- Avoid logging sensitive request payloads, credentials, or DB details.

# MCP Servers AGENTS.md - Local Codex Hints

This file applies only to `mcp_servers/`. Root `AGENTS.md` owns repo-wide rules.

## Server Map

These files expose standalone FastMCP servers from this repo:

- `oracle_knowledge.py`: typed evidence-only retrieval with `search_knowledge`,
  `list_knowledge_bases`, and `list_documents`; supports only `stdio` and
  `streamable-http`.

Do not confuse these standalone exposed servers with the app's MCP-consuming chat path. App chat consumes configured MCP servers through the LangGraph Agent Server `chat_agent` graph on port 2024, with `mode="mcp"` or `mode="mixed"` passed in run context.

## Where To Look

- `src/rag_agent/infrastructure/mcp_settings.py`: MCP client/server config resolution conventions.
- `src/rag_agent/infrastructure/mcp_adapter_runtime.py`: app-side MCP client wiring through LangChain's first-party `MCPAdapter` and FastMCP 4.
- `docs/MCP-USAGE.md`: expose-vs-consume behavior and local usage examples.
- `docs/CONFIGURATION.md`: MCP env vars, including server runtime settings.

## Runtime Config

- Standalone servers read settings through `api.settings.get_settings()`.
- Oracle Knowledge uses `ORACLE_KNOWLEDGE_TRANSPORT`, namespaced host/port, and the `/mcp` path.
- Consuming-side server selection belongs in the Settings UI config store (with env/config fallback for headless operation). Do not hardcode consumed server URLs in standalone server code.
- Never commit credentials, wallet paths, tokens, or environment-specific external endpoints.

## Run And Smoke Test

Oracle Knowledge MCP (friendly keys, no raw table names):

```bash
uv run python mcp_servers/oracle_knowledge.py  # stdio by default
ORACLE_KNOWLEDGE_TRANSPORT=streamable-http uv run python mcp_servers/oracle_knowledge.py
```

The HTTP profile is served at `/mcp`; use the dedicated
`docker-compose.oracle-knowledge.yml` for the isolated container. Logs go to
stderr so STDIO stdout remains MCP protocol clean. The server has no built-in
MCP authentication; use loopback or a trusted authenticated network boundary.

The FastAPI app does not need to be running to start these standalone servers, but the underlying retrieval/RAG dependencies still need valid local config. Check `docker compose ps` first and reuse any already-running database/backend support services when relevant.

## Boundaries

- Keep tools small, typed, and JSON-serializable.
- Keep standalone MCP tool implementations in `mcp_servers/*`; do not hide them in FastAPI handlers.
- Keep app-side MCP consumption in `src/rag_agent/infrastructure/mcp_adapter_runtime.py` and runtime orchestration layers.
- Update `docs/MCP-USAGE.md`, `docs/CONFIGURATION.md`, and deployment-profile tests when tool names, server profiles, or payload shapes change.
- Avoid logging sensitive request payloads, credentials, or DB details.

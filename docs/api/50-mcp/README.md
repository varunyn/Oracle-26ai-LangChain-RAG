# MCP API

## POST `/api/mcp/chat`

This route is a **legacy compatibility endpoint**. It remains in the FastAPI surface, but it is not the primary supported MCP integration path in the current repo state.

Use `POST /api/chat` with `mode="mcp"` or `mode="mixed"` for active MCP-enabled chat.

## Current repo behavior

`/api/mcp/chat` depends on the legacy `AgentWithMCP` path. In this repository that implementation is intentionally stubbed, so the endpoint currently reports:

```json
{
  "error": "MCP integration not available"
}
```

When `stream=true`, it returns the same error through SSE.

## Request fields

- `messages`
- `mcp_url` (optional)
- `stream` (defaults to true)

If `mcp_url` is not provided, the route attempts to use a configured default MCP server URL. That compatibility behavior is separate from the standalone MCP servers exposed by `mcp_servers/*`, which serve tools under `/mcp`.

## Supported alternative

Use `/api/chat` for the app's supported MCP path:

- `mode="mcp"` for MCP-only tool use
- `mode="mixed"` for retrieval with MCP fallback/tool routing
- `mcp_server_keys` to limit which configured MCP servers/tools are loaded

See `../20-chat/README.md` and `../../MCP-USAGE.md` for the active MCP flow.

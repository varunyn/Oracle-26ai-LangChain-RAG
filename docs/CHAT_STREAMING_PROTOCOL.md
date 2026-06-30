# Chat streaming protocol

This repo supports chat streaming through LangGraph Agent Server:

- `POST /threads/{thread_id}/runs/stream`
- `POST /threads/{thread_id}/stream`

## Streaming

The frontend uses `@langchain/react` against `assistantId: "chat_agent"`. It submits
standard `messages` input plus top-level runtime `context` and receives Agent
Server streaming updates.

Run-start request shape:

```json
{
  "assistant_id": "chat_agent",
  "input": {
    "messages": [{ "role": "user", "content": "Hello" }]
  },
  "context": {
    "mode": "rag"
  },
  "stream_mode": ["values", "messages", "tools"]
}
```

Notes:

- Stream completion is transport close (there is no `[DONE]` sentinel).
- Assistant references and sources are carried in `response_metadata` / `additional_kwargs` on assistant messages.
- Frontend uses `@langchain/react` directly against `NEXT_PUBLIC_LANGGRAPH_API_BASE` (local default: `http://localhost:2024`).
- FastAPI no longer adapts or proxies the chat protocol.

Example request:

```bash
curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"assistant_id":"chat_agent","input":{"messages":[{"role":"user","content":"Hello"}]},"context":{"mode":"direct"},"stream_mode":["values","messages","tools"]}' \
  http://localhost:2024/threads/thread-1/runs/stream

curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"assistant_id":"chat_agent","stream_mode":["values","messages","tools"]}' \
  http://localhost:2024/threads/thread-1/stream
```

## Server-owned memory: delta-only input + thread IDs

- The LangGraph Agent Server is the source of truth for conversation context and thread state.
- API requests should contain at least one user message in `input.messages`.
- `thread_id` is the conversation identifier.
  - Frontend persists only the active `thread_id` as a reopen pointer and reuses it on later turns.
  - Sidebar history is loaded from the Agent Server `threads.search(...)` response and is not persisted in browser storage.

### Tool activity channels

- Native Agent Server tool calls use the `tools` channel and are projected by `@langchain/react` as `stream.toolCalls`.
- MCP history/replay metadata is retained on the final assistant message in `mcp_tool_invocations`.

### Inspecting + deleting thread state

- Inspect through the Agent Server:

```bash
curl -s http://localhost:2024/threads/search
curl -s http://localhost:2024/threads/<thread_id>/state
```

- Delete via LangGraph client or Agent Server thread APIs.
- Reset all state: delete the thread through Agent Server APIs or clear the configured checkpoint storage.

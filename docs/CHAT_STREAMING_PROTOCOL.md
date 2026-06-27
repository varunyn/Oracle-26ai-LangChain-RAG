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
- Assistant references, sources, and tool activity are carried in `response_metadata` / `additional_kwargs` on assistant messages.
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

- The server is the source of truth for conversation context in `ChatRuntimeService` (`src/rag_agent/runtime/chat_service.py`).
- API requests should contain at least one user message in `input.messages`.
- `thread_id` is the conversation identifier.
  - Frontend persists `thread_id` in `localStorage` and reuses it on later turns.

### Inspecting + deleting thread state

- Inspect programmatically:

```bash
uv run python - <<'PY'
import asyncio
from api.dependencies import build_chat_config
from src.rag_agent.runtime.chat_service import ChatRuntimeService

async def main() -> None:
    run_config = build_chat_config(thread_id="t1")
    snap = await ChatRuntimeService().get_state(run_config)
    values = getattr(snap, "values", {}) or {}
    print(len(values.get("messages") or []))

asyncio.run(main())
PY
```

- Delete via LangGraph client or Agent Server thread APIs.
- Reset all state: restart the relevant process if you are using process-local memory.

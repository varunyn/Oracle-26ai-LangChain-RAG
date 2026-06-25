# Chat streaming protocol

This repo supports chat streaming on:

- `POST /api/langgraph/threads/{thread_id}/commands`
- `POST /api/langgraph/threads/{thread_id}/stream/events`

## Streaming

The frontend uses the `@langchain/react` v1 protocol. It submits a command as JSON,
then reads protocol events from the event stream endpoint.

Command request shape:

```json
{
  "id": 1,
  "method": "run.start",
  "params": {
    "assistant_id": "mcp_agent_executor",
    "input": {
      "messages": [{ "type": "human", "content": "Hello" }]
    }
  }
}
```

The command response is JSON:

```json
{
  "id": 1,
  "type": "success",
  "result": { "run_id": "generated-run-id", "created_at": null }
}
```

The event stream response is SSE with protocol events whose `method` is `values`,
`tools`, or `lifecycle`.

SSE framing:

```text
event: event
data: {"type":"event","seq":1,"event_id":"...","method":"values","params":{"namespace":[],"data":{"messages":[...]}}}

event: event
data: {"type":"event","seq":2,"event_id":"...","method":"lifecycle","params":{"namespace":[],"data":{"event":"completed"}}}
```

Notes:

- Stream completion is transport close (there is no `[DONE]` sentinel).
- Assistant references, sources, and tool activity are carried in `response_metadata` / `additional_kwargs` on assistant messages.
- Frontend uses `@langchain/react` `useStream` against `${NEXT_PUBLIC_API_BASE}/api/langgraph`.
- Internally, the command route consumes `ChatRuntimeService.astream_events(..., version="v3")`
  and adapts text/tool-call/reference projections into this values-stream contract.

Example request:

```bash
curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"id":1,"method":"run.start","params":{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"Hello"}]}}}' \
  http://localhost:3002/api/langgraph/threads/thread-1/commands

curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"channels":["values","tools","lifecycle"],"depth":1}' \
  http://localhost:3002/api/langgraph/threads/thread-1/stream/events
```

## Server-owned memory: delta-only input + thread IDs

- The server is the source of truth for conversation context in `ChatRuntimeService` (`src/rag_agent/runtime/chat_service.py`).
- API requests should contain at least one user/human message in `input.messages`.
- `thread_id` is the conversation identifier.
  - Frontend persists `thread_id` in `localStorage` and reuses it on later turns.
- Streaming contract uses `event: event` SSE frames with protocol event payloads.

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

- Delete via API: `DELETE /api/threads/{thread_id}` (idempotent 204)
- Reset all state: restart the API process (current memory store is process-local)

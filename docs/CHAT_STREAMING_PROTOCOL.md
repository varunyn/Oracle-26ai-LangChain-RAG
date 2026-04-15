# Chat streaming protocol

This repo supports chat streaming on:

- `POST /api/langgraph/threads/{thread_id}/runs/stream`

## Streaming

When streaming through thread/run endpoints, the response is SSE with `event: values` frames.

SSE framing:

```text
event: values
data: {"messages":[...]}

event: values
data: {"messages":[...]}
```

Notes:

- Stream completion is transport close (there is no `[DONE]` sentinel).
- Assistant references/citations are carried in `response_metadata` / `additional_kwargs`.
- Frontend uses `@langchain/react` `useStream` against `${NEXT_PUBLIC_API_BASE}/api/langgraph`.

Example request:

```bash
curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"Hello"}]}}' \
  http://localhost:3002/api/langgraph/threads/thread-1/runs/stream
```

## Non-streaming (`stream: false`)

When `stream: false`, use `POST /api/langgraph/threads/{thread_id}/runs` and read the `output` object.

Example request:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"Hello"}]}}' \
  http://localhost:3002/api/langgraph/threads/thread-1/runs
```

---

## Server-owned memory: delta-only input + thread IDs

- The server is the source of truth for conversation context in `ChatRuntimeService` (`api/services/graph_service.py`).
- API requests should contain at least one user/human message in `input.messages`.
- `thread_id` is the conversation identifier.
  - Frontend persists `thread_id` in `localStorage` and reuses it on later turns.
- Streaming contract uses `event: values` SSE frames.

### Inspecting + deleting thread state

- Inspect programmatically:

```bash
uv run python - <<'PY'
import asyncio
from api.dependencies import build_chat_config
from api.services.graph_service import ChatRuntimeService

async def main() -> None:
    run_config = build_chat_config(thread_id="t1")
    snap = await ChatRuntimeService().get_state(run_config)
    values = getattr(snap, "values", {}) or {}
    print(len(values.get("messages") or []))

asyncio.run(main())
PY
```

- Delete via API: `DELETE /api/threads/{thread_id}` (204 on success, 404 if missing)
- Reset all state: restart the API process (current memory store is process-local)

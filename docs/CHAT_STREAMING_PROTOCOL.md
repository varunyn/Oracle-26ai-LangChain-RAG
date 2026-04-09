# Chat streaming protocol

This repo supports chat streaming on:

- `POST /api/chat`

## Streaming (`stream: true`)

When `stream: true`, the response streams using the **AI SDK UI Message Stream Protocol**.

Required response header:

```http
x-vercel-ai-ui-message-stream: v1
```

SSE framing:

```text
data: <json>\n\n
...
data: [DONE]\n\n
```

Frontend proxy note:

- The Next.js route `frontend/src/app/api/chat/route.ts` proxies the upstream stream.
- It must not parse SSE or buffer/transform the stream.

Example request:

```bash
curl -sS -N \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}' \
  http://localhost:3002/api/chat
```

## Non-streaming (`stream: false`)

When `stream: false`, the response is a single JSON payload. The JSON shape is unchanged for now.

Example request:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  http://localhost:3002/api/chat
```

---

## Server-owned memory: delta-only input + thread IDs

- The server is the source of truth for chat history using a LangGraph checkpointer (SQLite by default) and the canonical `messages` state with the `add_messages` reducer.
- API requests MUST be delta-only: `messages` must contain EXACTLY ONE `{ role: "user", content: ... }` per request. The graph restores prior messages from persistence.
- `thread_id` is the conversation identifier.
  - If omitted: server generates and uses a new `thread_id` (returned in non-stream JSON responses).
  - The Next.js UI persists `thread_id` in `localStorage` and reuses it on subsequent turns.
- Streaming contract is unchanged: same headers, SSE frames, and `data: [DONE]` terminator.

### Inspecting + deleting memory

- Inspect programmatically:

```bash
uv run python - <<'PY'
from api.dependencies import build_chat_config
from api.services.graph_service import GraphService
snap = GraphService().get_state(build_chat_config(thread_id="t1"))
vals = getattr(snap, "values", {}) or {}
print(len(vals.get("messages") or []))
PY
```

- Delete via API: `DELETE /api/threads/{thread_id}` (204 on success, 404 if missing)
- Reset all memory: stop the server and remove the SQLite file (`.local-data/langgraph-checkpoints.sqlite` or `$LANGGRAPH_SQLITE_PATH`), then restart

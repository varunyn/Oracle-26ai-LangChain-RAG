# Server-owned conversation memory

This app keeps chat memory on the backend in `ChatRuntimeService` (`src/rag_agent/runtime/chat_service.py`). The client sends only the latest user turn (delta-only), and the service stores thread-scoped state under `thread_id`.

## Key behaviors

- API contract: `POST /api/langgraph/threads/{thread_id}/commands` plus
  `POST /api/langgraph/threads/{thread_id}/stream/events`
  - Request input should include at least one latest user/human message.
  - `thread_id` identifies a conversation; create one with `POST /api/langgraph/threads`.
  - Stream responses emit `event: event` SSE protocol frames.
- Storage model
  - Runtime always keeps a process-local cache (`self._thread_state`).
  - When `ENABLE_PERSISTENT_MEMORY=true`, state is also stored in the SQLite checkpoint file configured by `LANGGRAPH_SQLITE_PATH`.
  - State includes normalized LangChain messages and the last answer metadata.
  - `DELETE /api/threads/{thread_id}` removes a conversation state entry and is idempotent.

## Scope and limitations

- Memory is thread-scoped.
- With default settings, memory is not durable across process restarts.
- With `ENABLE_PERSISTENT_MEMORY=true`, memory survives API restarts through the configured SQLite checkpoint file.
- Memory is not shared across multiple API replicas unless they share the same durable state backend.

## Inspect thread memory

```bash
uv run python - <<'PY'
import asyncio
from api.dependencies import build_chat_config
from src.rag_agent.runtime.chat_service import ChatRuntimeService

THREAD_ID = "example-thread"

async def main() -> None:
    svc = ChatRuntimeService()
    snapshot = await svc.get_state(build_chat_config(thread_id=THREAD_ID))
    values = getattr(snapshot, "values", {}) or {}
    print(f"thread_id={THREAD_ID} messages_count={len(values.get('messages') or [])}")

asyncio.run(main())
PY
```

## Delete one thread

- API: `DELETE /api/threads/{thread_id}` returns 204 whether or not the thread already exists.
- Programmatic: `await ChatRuntimeService().delete_thread("<thread_id>")`

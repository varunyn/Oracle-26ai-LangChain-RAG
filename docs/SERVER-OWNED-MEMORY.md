# Server-owned conversation memory

This app keeps chat memory on the backend in `ChatRuntimeService` (`api/services/graph_service.py`). The client sends only the latest user turn (delta-only), and the service stores thread-scoped state under `thread_id`.

## Key behaviors

- API contract: `/api/langgraph/threads/{thread_id}/runs`
  - Request input should include at least one latest user/human message.
  - `thread_id` identifies a conversation; if missing, the server generates one.
  - Non-stream responses include top-level `thread_id`.
- Storage model
  - Current implementation uses an in-memory per-process map (`self._thread_state`).
  - State includes normalized LangChain messages and the last answer metadata.
  - `DELETE /api/threads/{thread_id}` removes a conversation state entry.

## Scope and limitations

- Memory is short-term and thread-scoped.
- Memory is not durable across process restarts.
- Memory is not shared across multiple API replicas.

## Inspect thread memory

```bash
uv run python - <<'PY'
import asyncio
from api.dependencies import build_chat_config
from api.services.graph_service import ChatRuntimeService

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

- API: `DELETE /api/threads/{thread_id}` returns 204 when thread state exists.
- Programmatic: `await ChatRuntimeService().delete_thread("<thread_id>")`

# Server-owned conversation memory

This app keeps chat memory on the backend in `ChatRuntimeService` (`src/rag_agent/runtime/chat_service.py`). The client sends only the latest user turn (delta-only), and the service stores thread-scoped state under `thread_id`.

## Key behaviors

- Agent Server contract: thread/run/stream endpoints on the LangGraph Agent Server
  - Request input should include at least one latest user/human message.
  - `thread_id` identifies a conversation and is created/managed by the Agent Server.
- Storage model
  - Runtime always keeps a process-local cache (`self._thread_state`).
  - When `ENABLE_PERSISTENT_MEMORY=true`, state is also stored in the SQLite checkpoint file configured by `LANGGRAPH_SQLITE_PATH`.
  - State includes normalized LangChain messages and the last answer metadata.
  - Thread deletion is now done through Agent Server client APIs.

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

- Programmatic runtime cleanup: `await ChatRuntimeService().delete_thread("<thread_id>")`
- Product UI cleanup uses the LangGraph client thread delete API.

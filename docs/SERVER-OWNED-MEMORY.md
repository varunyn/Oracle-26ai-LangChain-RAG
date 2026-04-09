# Server-owned conversation memory (LangGraph)

This app persists chat conversations using LangGraph "messages" state plus a durable checkpointer. The server is the source of truth for memory; clients send only the new user message (delta-only), and the graph appends the new assistant message to the persisted history. For session vs thread, single-thread UX, and memory scope, see [CHAT_MEMORY_AND_SESSIONS.md](./CHAT_MEMORY_AND_SESSIONS.md).

## Why this change?

- Prevents history duplication and payload bloat
- Matches LangGraph best practice: `messages: Annotated[list[AnyMessage], add_messages]`
- Durable across restarts and between turns using a DB-backed checkpointer (SQLite by default)

## Key behaviors

- API contract: `/api/chat`
  - Delta-only input: request `messages` must contain EXACTLY 1 `{ role: "user", content: ... }`
  - Streaming protocol (AI SDK UI Message Stream) is unchanged: same headers, SSE frames, and terminator
  - `thread_id` identifies the conversation. If omitted, the server generates one. Non-stream JSON responses include `thread_id` for the caller to persist.
- LangGraph state
  - Canonical field: `messages` with `add_messages` reducer
- Nodes read from `messages`; `Search`, `AnswerFromDocs`, and `DraftAnswer` can use conversation history, and `DraftAnswer` appends the assistant `AIMessage`
  - Persisted `messages` are trimmed to the most recent `MAX_MSGS_IN_HISTORY` entries before checkpoint write
  - Checkpointer stores state keyed by thread_id

## Storage location (SQLite)

By default, a single SQLite file holds all conversation memory:

1. `LANGGRAPH_SQLITE_PATH` environment variable (if set)
2. Otherwise: `<repo-root>/.local-data/langgraph-checkpoints.sqlite`

Code: `src/rag_agent/langgraph/graph.py` (see `_default_sqlite_path()` and `get_default_checkpointer()`).

## Config knobs (set in `.env` or environment)

See `.env.example` and `docs/CONFIGURATION.md`. Example:

- `ENABLE_PERSISTENT_MEMORY = True`
  - If `False`, wire the graph to use an in-memory saver (ephemeral, not persisted)
- `LANGGRAPH_SQLITE_PATH = None`
  - Optional explicit DB path, e.g. `/var/app/data/langgraph-checkpoints.sqlite`
  - If `None`, env `LANGGRAPH_SQLITE_PATH` or repo default is used
- `ALLOW_CLIENT_THREAD_ID = True`
  - If `False`, server always generates/uses its own `thread_id` and ignores client-provided IDs
- `THREAD_ID_STRATEGY = "uuid4"` (or `"random-hex-8"` if wired)
- `THREAD_ID_PREFIX = ""` (prepend a fixed prefix when generating IDs)

Notes:

- Today the code honors the env-var path and server generation; wiring `ENABLE_PERSISTENT_MEMORY`, `LANGGRAPH_SQLITE_PATH` (config override), and thread ID strategy/prefix is straightforward if you want server-only control. Ask to "wire it" and specify defaults.

## Client thread_id generation

Frontend (Next.js) behavior:

- `generateThreadId()` uses `crypto.randomUUID()` (fallbacks to `thread-<ts>-<rand>`)
- Saved in `localStorage` under key `rag_agent_thread_id`
- Passed to the API via the Next.js `/api/chat` route proxy

If you prefer server-only IDs, either:

- Set `ALLOW_CLIENT_THREAD_ID=False` and adjust the UI to mint or fetch a server thread ID first, or
- Keep the current client UUIDs (common and acceptable)

## Inspect persisted memory (programmatic)

```bash
uv run python - <<'PY'
from api.dependencies import build_chat_config
from api.services.graph_service import GraphService

THREAD_ID = "t1"
run_config = build_chat_config(thread_id=THREAD_ID)
snap = GraphService().get_state(run_config)
vals = getattr(snap, "values", {}) or {}
msgs = vals.get("messages") or []
print(f"thread_id={THREAD_ID} messages_count={len(msgs)}")
for i, m in enumerate(msgs, 1):
    print(i, type(m).__name__, getattr(m, "content", None))
PY
```

## Delete conversation memory (single thread)

- API: `DELETE /api/threads/{thread_id}` → 204 on success, 404 if missing
- Programmatic: `GraphService().delete_thread("<thread_id>")`

## Reset ALL memory (dangerous)

- Stop the API server, then remove the SQLite file and restart:

```bash
rm -f .local-data/langgraph-checkpoints.sqlite
# or: rm -f "$LANGGRAPH_SQLITE_PATH" if you set it
```

## Testing and verification

- Delta-only validation: sending 0 or >1 messages returns a 422 error
- Persistence tests: two turns with the same `thread_id` produce 4 messages (Human+AI per turn)
- Delete endpoint tests: first delete → 204, second delete → 404; state is empty afterward
- Streaming smoke: `/api/chat` SSE protocol unchanged

See `tests/test_chat_persistence_and_delete.py` and `tests/test_openapi_baseline.py`.

## Troubleshooting

- The app uses **AsyncSqliteSaver** (created in FastAPI lifespan) so `graph.astream()` and checkpoint access work natively in async. If the lifespan did not run (e.g. some test clients), `GraphService` falls back to sync `get_state`/`delete_thread` when the checkpointer raises for async methods; `astream` still has a thread fallback for sync-only checkpointers.
- If memory seems to duplicate:
  - Ensure your request sends only the new user message (delta-only). With a checkpointer, the graph restores prior messages automatically.

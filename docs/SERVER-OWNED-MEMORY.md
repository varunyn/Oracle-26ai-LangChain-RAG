# Server-owned conversation memory

This app keeps chat memory on the backend in the LangGraph Agent Server thread/checkpoint layer. The client sends only the latest user turn (delta-only), and server-owned thread state is keyed by `thread_id`.

## Key behaviors

- Agent Server contract: thread/run/stream endpoints on the LangGraph Agent Server
  - Request input should include at least one latest user/human message.
  - `thread_id` identifies a conversation and is created/managed by the Agent Server.
- Storage model
  - The Agent Server owns thread state and replay.
  - Graph checkpoints are stored in the SQLite checkpoint file configured by `LANGGRAPH_SQLITE_PATH`.
  - In local Docker/dev mode, the Agent Server thread/run registry is stored in `.langgraph_api`.
  - State includes LangChain messages plus assistant metadata persisted by the graph.
  - Thread deletion is done through Agent Server client APIs.

## Scope and limitations

- Memory is thread-scoped.
- Memory survives container recreation when both `./local-data` and `./.langgraph_api` are mounted.
- Memory is not shared across multiple API replicas unless they share the same durable state backend.

## Inspect thread memory

```bash
curl -s http://localhost:2024/threads/search
```

## Delete one thread

- Product UI cleanup uses the LangGraph client thread delete API.

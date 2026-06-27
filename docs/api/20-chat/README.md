# Chat API

Chat execution no longer uses FastAPI `/api/langgraph/*` compatibility routes.

The end state is the LangGraph Agent Server graph `chat_agent`, typically served
locally from `http://127.0.0.1:2024`.

## What FastAPI still owns

The FastAPI app continues to serve product endpoints such as:

- `/api/config`
- `/api/suggestions`
- `/api/feedback`
- `/api/documents/*`

## What Agent Server owns

Chat threads, runs, state, history, and streaming now come directly from the
LangGraph Agent Server:

- `POST /threads`
- `POST /threads/{thread_id}/runs/stream`
- `POST /threads/{thread_id}/stream`
- `GET /threads/{thread_id}/state`
- `POST /threads/search`

Use the LangGraph base URL instead of the FastAPI base URL when exercising chat
manually or from frontend integrations.

## Frontend contract

The frontend uses `@langchain/react` with:

- `assistantId: "chat_agent"`
- standard `messages` graph input plus top-level runtime `context`
- server-backed thread history via `stream.client.threads.search(...)`

## Recommended local workflow

```bash
uv run langgraph dev
./run_api.sh
```

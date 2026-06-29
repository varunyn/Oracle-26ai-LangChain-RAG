# Chat memory and sessions

This doc describes conversation memory, **sessions**, and **threads** in the current runtime.

## Overview

| Concept | What it is | Where it lives | Used for |
| --- | --- | --- | --- |
| **Session** | One browser visit (new per tab load/refresh). | Frontend in-memory; sent as `session_id`. | Grouping Langfuse traces for a visit. |
| **Thread** | One conversation identifier. | Agent Server persistence/SQLite; frontend stores only the active ID pointer. | Server-side short-term memory lookup. |

## Runtime memory model

- Backend memory is owned by the LangGraph Agent Server graph/checkpointer.
- With persistent checkpointer storage enabled, thread state is stored in the SQLite checkpoint file configured by `LANGGRAPH_SQLITE_PATH`.
- Input is delta-only: each request should send the new user message.
- Memory survives across turns in one process. With the default settings it is cleared on process restart; with persistent memory enabled it is rehydrated from SQLite.

## Session vs thread

- `session_id`
  - New per tab load/refresh.
  - Not persisted in browser storage.
  - Used for observability correlation.
- `thread_id`
  - The authoritative thread state and history live in the Agent Server persistence layer/SQLite.
  - The active ID is persisted in browser `localStorage` (`rag_agent_thread_id`) only so the current conversation can be reopened after a reload.
  - Sent with each chat request for continuity.
  - Cleared/replaced when user triggers “Clear chat”.

The sidebar is an in-memory projection of `threads.search(...)`. It is replaced by server results and does not persist conversation summaries in browser storage.

## Where it’s implemented

- Session/thread handling: `frontend/src/hooks/useChatSession.ts`
- Chat request schema: `api/schemas.py`
- Chat runtime state: `src/rag_agent/graphs/chat_agent.py`
- Thread delete endpoint: LangGraph Agent Server (`DELETE /threads/{thread_id}`)
- Langfuse metadata injection: `src/rag_agent/utils/langfuse_tracing.py`

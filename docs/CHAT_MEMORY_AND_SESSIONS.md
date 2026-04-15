# Chat memory and sessions

This doc describes conversation memory, **sessions**, and **threads** in the current runtime.

## Overview

| Concept | What it is | Where it lives | Used for |
| --- | --- | --- | --- |
| **Session** | One browser visit (new per tab load/refresh). | Frontend in-memory; sent as `session_id`. | Grouping Langfuse traces for a visit. |
| **Thread** | One conversation identifier. | Frontend `localStorage` + backend request body as `thread_id`. | Server-side short-term memory lookup. |

## Runtime memory model

- Backend memory is owned by `ChatRuntimeService` (`api/services/graph_service.py`).
- The service stores thread state in a process-local dictionary.
- Input is delta-only: each request should send the new user message.
- Memory survives across turns in one process, and is cleared on process restart.

## Session vs thread

- `session_id`
  - New per tab load/refresh.
  - Not persisted in browser storage.
  - Used for observability correlation.
- `thread_id`
  - Persisted in browser `localStorage` (`rag_agent_thread_id`).
  - Sent with each chat request for continuity.
  - Cleared/replaced when user triggers “Clear chat”.

## Where it’s implemented

- Session/thread handling: `frontend/src/hooks/useChatSession.ts`
- Chat request schema: `api/schemas.py`
- Chat runtime state: `api/services/graph_service.py`
- Thread delete endpoint: `src/rag_agent/runtime/langgraph_server.py` (`DELETE /api/threads/{thread_id}`)
- Langfuse metadata injection: `src/rag_agent/utils/langfuse_tracing.py`

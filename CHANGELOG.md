# Changelog

## 2026-06-27

- Removed the unused `graph` constructor argument from `ChatRuntimeService` and updated the remaining unit tests that were still passing it as dead compatibility baggage.
- Removed the unused `ChatRuntimeService.delete_thread()` helper, renamed the runtime docstrings to reflect shared LangGraph/FastAPI ownership, and added section markers in `chat_service.py` to make future cleanup safer.
- Restored LangGraph-to-Langfuse session propagation by carrying `session_id` through the Agent Server context schema into graph node runtime calls, so LangGraph chat traces keep the same session correlation as the FastAPI runtime path.
- Fixed follow-up suggestions for OCI Grok-selected chats by falling back to the default suggestions model when LangChain structured output passes unsupported `strict` kwargs through `langchain-oci`.
- Fixed LangGraph graph nodes to return a terminal assistant error message when the runtime backend raises, so `@langchain/react` streams stop loading instead of leaving the UI on “Preparing response” after retrieval/backend failures.
- Updated Docker and local frontend defaults to use `NEXT_PUBLIC_LANGGRAPH_API_BASE=http://localhost:2024` for direct browser-to-Agent-Server streaming, and kept LangGraph stream ownership inside the active `@langchain/react` controller when new thread ids are assigned.
- Configured LangGraph Agent Server CORS for local frontend origins so browser-direct thread search and streaming requests can call `http://localhost:2024`.
- Mounted `langgraph.json` and set `CORS_ALLOW_ORIGINS` on the Docker LangGraph service so local Agent Server CORS/config changes apply with a container recreate instead of requiring an image rebuild.

## 2026-06-25

- Migrated LangGraph chat streaming to the `@langchain/react` v1 command and protocol event endpoints.
- Updated LangChain/LangGraph-related runtime dependencies and removed the local `langchain-oracle` Docker build override.
- Refreshed API docs, Bruno requests, and OpenAPI fixtures for `/commands` and `/stream/events`.
- Fixed frontend chat rendering so submitted suggestion prompts are not rendered twice.

## 2026-06-26

- Rewired the frontend chat client to LangGraph Agent Server conventions with `@langchain/react`, `assistantId: "chat_agent"`, standard `messages + context` submit payloads, and server-backed thread history from `threads.search(...)`.
- Removed the FastAPI `/api/langgraph/*` compatibility router and converted the local chat Bruno examples to external LangGraph Agent Server requests.
- Split Playwright coverage into deterministic mocked chat-streaming tests and live-backend RAG chat tests.
- Switched the Playwright frontend web server default port to `4040` to keep e2e runs separate from the normal frontend dev port.
- Aligned LangGraph protocol user-message ids between frontend optimistic values and backend current-turn values so submitted questions render once and before the assistant response.
- Added the initial LangGraph Agent Server bootstrap surface with `langgraph.json` and a minimal `chat_agent` graph while the legacy chat surface still coexists during the compatibility phase.
- Fixed LangGraph direct/RAG routing to use runtime context instead of graph state, reject unsupported `mcp`/`mixed` modes explicitly, move blocking RAG retrieval work off the Agent Server event loop, and cover the graph-mode contract with deterministic workflow tests plus real SDK integration tests.
- Added LangGraph graph-owned `mcp` and `mixed` routes, including live MCP/calculator and mixed retrieval-plus-tool verification against the Agent Server.
- Fixed LangGraph MCP config resolution to load the UI-managed MCP server store from the repo root during Agent Server runs.
- Added a first-class `langgraph` Docker Compose service, updated the core Docker stack to start backend + LangGraph + frontend together, and pointed the containerized frontend at the internal `http://langgraph:2024` service by default.
- Pinned the local Agent Server runtime to `langgraph-api>=0.10,<0.11` so the Docker LangGraph service matches the `@langchain/react` event-streaming protocol used by the frontend.
- Fixed the LangGraph chat graph to use standard Agent Server message state: graph nodes now emit `AIMessage` outputs with attached reference metadata, LangChain content blocks stay structured instead of being stringified, and persistence remains server-owned per LangGraph Agent Server conventions.
- Fixed the frontend LangGraph submit path to pass mode, collection, model, tracing, and reranker settings as top-level run `context` instead of embedding them inside graph input state, so live Agent Server runs route to the selected direct/RAG/MCP/mixed mode.

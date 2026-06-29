# Changelog

## 2026-06-29

- Updated root and scoped `AGENTS.md` guidance to match the active LangGraph Agent Server chat architecture, UI-managed MCP configuration, current Docker/frontend/docs-site commands, and repository testing boundaries.
- Fixed LangGraph thread-state message serialization to preserve structured assistant content blocks instead of stringifying them into Python-list text, so multi-turn chats keep the latest message parseable by the frontend while history remains server-owned.
- Added frontend compatibility parsing for older LangGraph thread turns whose assistant content was already persisted as stringified content-block lists, so existing chats render readable text without clearing the thread.
- Deduplicated replayed LangGraph stream messages by stable message id in the frontend projection layer so second-turn runs do not render already-persisted assistant messages a second time.
- Added development-only frontend chat debug logging around submit, projection, and render stages so duplicated message replays can be traced with message ids and compact previews instead of inferred from screenshots.
- Recovered automatically from stale persisted LangGraph thread ids by clearing missing-thread hydration state instead of leaving the frontend chat stuck on a 404 thread error.
- Rebased frontend chat rendering on authoritative LangGraph `values.messages` and collapsed adjacent duplicate assistant variants, so a single LangGraph run no longer renders the same final answer twice when streamed and persisted message ids differ.
- Fixed the LangGraph chat message reducer to merge streamed turns by stable message id instead of `(role, content)`, preventing one assistant response from being appended repeatedly as its streamed content grows during a single run.
- Tagged internal LangGraph Python LLM calls with `nostream` so per-call model streams no longer surface as duplicate visible chat messages while final assistant turns continue to come from graph state.
- Simplified the frontend chat transcript path so `stream.messages` is now the only visible-message source, removed raw-state/legacy content parsing fallbacks, and kept citations, durable threads, retries, replay deduplication, and MCP tool progress covered by focused frontend tests.
- Moved LangGraph MCP/mixed workflow-policy and retrieval-decision helpers into a graph-owned `mcp_policies` module, updated graph nodes to import from it directly, and removed that LangGraph-only policy code from `ChatRuntimeService`.
- Reduced `ChatRuntimeService` to a direct/RAG compatibility adapter, made it reject LangGraph-owned `mcp` and `mixed` modes, and migrated the remaining unit/integration assertions for those modes onto the graph and MCP-turn surfaces.
- Removed unused FastAPI ownership of `ChatRuntimeService`; app/request resources now keep only product-API settings and optional durable state while LangGraph remains the chat runtime surface.
- Moved LangGraph `mcp` and `mixed` chat execution into graph-owned nodes, added node-level tests for MCP/retrieval orchestration, and updated workflow coverage so the Agent Server path no longer depends on `ChatRuntimeService` for those modes.
- Split `ChatRuntimeService.run_chat()` into explicit private direct, RAG, MCP, and mixed-mode helpers, then removed the extra wrapper-helper layer and private-structure tests so the cleanup reduces code instead of adding new indirection.

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

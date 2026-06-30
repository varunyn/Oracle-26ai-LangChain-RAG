# Changelog

## 2026-06-30

- Added per-thread delete actions in the chat sidebar, reusing the LangGraph Agent Server thread delete path for both non-active thread removal and active-thread clear/delete behavior.
- Stabilized sidebar thread titles so selecting an existing thread no longer rewrites fallback labels from a random thread suffix to the first user question during replay.
- Fixed frontend sidebar thread-history reordering on old-thread selection by keeping existing thread title updates metadata-only instead of rewriting local recency timestamps, so backend `updated_at` ordering from LangGraph/SQLite remains authoritative.
- Fixed local Docker chat persistence after browser refresh/container recreation by bind-mounting the LangGraph dev runtime `.langgraph_api` thread/run registry alongside the SQLite checkpoint directory.
- Fixed OCI Gemini mixed/MCP tool calls by stripping Gemini-rejected JSON Schema bounds (`exclusiveMinimum` / `exclusiveMaximum`) from provider-bound function declarations while keeping the standard `langchain_oci` `ChatOCIGenAI` path.
- Added a LangGraph Agent Server custom SQLite checkpointer backed by `LANGGRAPH_SQLITE_PATH` with a local checkpoint TTL, and removed the obsolete FastAPI-side checkpoint resource wrapper so chat thread persistence has one owner.
- Removed the standalone FastAPI backend Docker service and made the LangGraph Agent Server the default host for both chat streaming and custom FastAPI product routes served through `langgraph.json` `http.app`.
- Added local Langfuse storage controls: SDK-side OTEL attribute truncation, low-sample local `.env.example` defaults, MinIO lifecycle rules for event/media/export objects, Redis memory caps, and Docker log rotation for Langfuse services so self-hosted dev traces do not fill Docker volumes as quickly.
- Fixed follow-up suggestions generation to use LangChain `ToolStrategy` explicitly instead of first attempting provider-native structured output with the selected model and falling back after `strict` provider errors, eliminating duplicate Langfuse `suggestions` traces.
- Fixed graph chat runs to finalize the root Langfuse `chat-*` observation with the latest user question and final answer. Live verification showed the root observation and child LangChain/LangGraph observations are now correct, while the self-hosted v3 trace list may still display child generation I/O.
- Fixed mixed-mode retrieval tool turns to keep the agent’s native final answer and attach RAG references to it instead of running a second RAG synthesis pass, preventing duplicate payment-terms answers after `oracle_retrieval`.
- Fixed native MCP tool-call rendering in the chat UI by normalizing the current `@langchain/react` `stream.toolCalls` shapes before matching them to assistant `tool_calls` ids, without restoring message-derived or MCP activity fallbacks.
- Registered LangGraph `ToolCallTransformer` on the compiled `chat_agent` and stopped returning inner MCP input humans to outer graph state, so MCP/mixed runs emit native `stream.toolCalls` without duplicating the submitted user message.
- Updated local Langfuse troubleshooting guidance to avoid the v4-only Observations API against the self-hosted v3 stack, using `traces get --fields ...observations...` as the supported local CLI path.
- Refactored graph-owned MCP/mixed execution to preserve the Agent Server runnable config and run streamed MCP tool execution through LangGraph `ToolNode`, so native tool lifecycle can surface through the LangGraph stream instead of a message/progress fallback.
- Removed the legacy MCP executor callback/projection fallback API (`tool_progress_callback`, `answer_delta_callback`, stop-after-tool handling, and tool-name server inference), leaving MCP/mixed tool lifecycle owned by the LangGraph stream path.
- Removed the legacy `ChatRuntimeService` and synthetic `v3_raw_event` compatibility stream adapter, along with their stale tests/docs, so chat execution, event streaming, and thread state are described and implemented only through the LangGraph Agent Server `chat_agent` path.
- Removed the frontend-only `custom:mcp_tool_activity` adapter and replay fallback so live tool UI now relies solely on native `@langchain/react` `stream.toolCalls`, simplifying the LangGraph chat contract around one authoritative tool-progress surface.
- Fixed LangGraph chat run-config construction to attach Langfuse LangChain callbacks for traced graph runs, so graph-owned `direct`, `rag`, `mcp`, and `mixed` chats can emit nested model/tool observations instead of only a root `chat-*` trace.
- Confirmed the local self-hosted Langfuse Docker stack is still pinned to `langfuse:3` / `langfuse-worker:3` under `observability/langfuse/docker-compose.yml`; this explains why the Observations v2 CLI endpoint is unavailable locally and should be upgraded separately from the runtime trace-wiring fix.

## 2026-06-29

- Fixed LangGraph Langfuse callback wiring to bind the callback handler to the configured project key explicitly instead of relying on ambient default-client resolution, preventing Docker `localhost:3300` fallback OTEL exports during traced MCP/mixed runs.

- Clarified the chat stream contract: native message/state/tool projections retain their documented meanings, MCP progress uses one custom channel, and configured MCP server identity is propagated explicitly instead of inferred from arbitrary tool names.

- Clarified frontend ownership so `@langchain/react` is the only chat runtime, `@langchain/core` is limited to message models/types, and AI Elements remains presentation-only.
- Added a canonical `custom:mcp_tool_activity` stream channel so internal MCP tool execution appears live in the UI without being misrepresented as native Agent Server tool calls.
- Removed persisted browser sidebar thread history so LangGraph/SQLite server state is the sole conversation-history source; localStorage now keeps only the active thread ID pointer.
- Fixed chat-history deletion so the active thread is removed from the sidebar only after the Agent Server DELETE succeeds, and a cleared unbound chat no longer refreshes and re-adds that thread.
- Added a mixed-mode progress state update so the SSE stream communicates retrieval/tool work before emitting one final citation-bearing assistant message.
- Added the matching RAG-mode progress state update so retrieval runs expose explicit progress before the single citation-bearing answer.
- Fixed live-to-final chat projection so the completed LangGraph state, including citation metadata, replaces the token-stream snapshot before rendering the finished answer.
- Fixed citation hydration precedence so the current run’s finalized `stream.values.messages` is used before an earlier thread-state snapshot, preventing sources from appearing only after refresh.
- Fixed citation rendering for serialized LangGraph state messages by recognizing `type: "ai"` and `type: "human"` alongside LangChain message instances.
- Hydrated the authoritative LangGraph thread state when a stream completes, so citation metadata is available immediately instead of only after a page refresh.
- Replaced the custom chat scroll hook with a local AI Elements `Conversation` source component, moved `ChatMessageList` into the Conversation-managed viewport, and added browser coverage for auto-scroll plus return-to-latest behavior without changing message, tool-call, or citation rendering.
- Added a repo-local `langgraph-chat-contract-debugging` skill that codifies how to separate backend truth, frontend projection bugs, and stale Playwright expectations when chat streaming, citations, tool calls, or mode-specific e2e failures disagree.
- Removed the misleading frontend `DOCUMENT_CHUNKS_VS` collection fallback, derived collection selection from the real `/api/config` payload, and showed an unavailable state instead when app config is missing.
- Replaced the frontend MCP progress translation layer with native `@langchain/react` `stream.toolCalls` rendering, preserved assistant `tool_calls` ids during message projection, and rendered per-message AI Elements tool cards directly from live `AssembledToolCall` state.
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

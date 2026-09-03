# Changelog

## 0.4.0 - 2026-09-03

- Migrated MCP consumption to LangChain 1.4's first-party `langchain.mcp.MCPAdapter` on FastMCP 4 and MCP SDK 2. The runtime now uses FastMCP `ClientGroup` namespacing, current snake_case protocol fields, explicit client cleanup, and the existing successful-result warning policy without the retired `langchain-mcp-adapters` callback and interceptor layer.
- Updated Oracle Knowledge MCP compatibility coverage for FastMCP 4.0.2, MCP SDK 2.1.1, and protocol version `2026-07-28`, including a real STDIO call through the app's LangChain adapter path.
- Removed the retired global MCP JWT/OAuth overlay, obsolete `mcp_use` logging and warning configuration, legacy SSE/underscore transport compatibility, and pre-FastMCP 4 connection fields. Per-server bearer and OAuth configuration is now the only MCP authentication path, and an explicitly empty runtime server override no longer falls back to saved servers.

## 2026-08-31

- Corrected the local Oracle Knowledge MCP environment and Codex STDIO configuration to use `ORACLE_WEB_EMBEDDINGS` through the `default` friendly key, removed unused `ORACLE_DB_*` forwarding, and documented every server setting plus its OCI embedding and Oracle Vector prerequisites.
- Raised the local Langfuse ClickHouse defaults to 2 CPUs and 8 GB of memory and documented the required Docker/Colima VM headroom, preventing background merge retry loops under the former 1 CPU/2 GB cap.
- Upgraded the self-hosted Langfuse web and worker images from 4.1.0 to 4.25.0 and the Python SDK from 4.14.1 to 4.14.4. OpenTelemetry resources now carry the configured deployment environment so LangSmith-exported spans no longer appear in Langfuse's `default` environment.

## 0.3.0 - 2026-08-21

- Replaced the legacy standalone semantic-search and full-RAG MCP contracts with the independent Oracle Knowledge MCP exposing `search_knowledge`, `list_knowledge_bases`, and `list_documents`.
- Added friendly-key allowlisting, typed outcomes, bounded retrieval/reranking, Streamable HTTP and STDIO deployment profiles, readiness probes, safe observability, and deployment-profile validation.
- Removed legacy `semantic_search`, collection-listing, `list_documents_in_collection`, `rag_ask`, code-mode, and manual-only MCP paths.
- Fixed live Oracle Vector retrieval to omit empty metadata filters instead of sending `filter={}`, which the provider interprets as a filtering expression and can turn valid searches into false no-hit results.
- Applied the Oracle Knowledge-specific reranker setting and hardened public MCP/chat failure boundaries so rejected collection identifiers and provider exception details cannot reach tool results, evidence artifacts, telemetry, or model-visible error text.
- Updated the release streaming smoke test to create and clean up a valid UUID Agent Server thread and to accept CRLF-terminated SSE event lines.

## 2026-08-20

- Upgraded the LangGraph Agent Server runtime from `langgraph-api` 0.10.0 to 0.12.6, including its resolved in-memory runtime and command-line dependencies. Aligned the directly pinned OpenTelemetry packages to the API-required 1.37/0.58 series and updated logging imports for that SDK's `LogData` handler interface.
- Deepened shared MCP/mixed tool-agent execution into `tool_agent_execution`, preserving native frontend tool streaming, persisted thread history, graph node names, and current chat result contracts. Added typed transient tool-execution transcript coverage for normal, failed, and incomplete tool sequences.
- Scoped MCP and mixed-mode transcript analysis to the latest user turn, preventing prior tool activity and answers from affecting current-turn policy, result metadata, or synthesis decisions.
- Treat unmatched tool calls as explicit incomplete-execution failures, so MCP outcome policy no longer silently drops an interrupted tool invocation.
- Preserve original tool-call order in MCP invocation metadata when completed and incomplete calls are mixed.
- Added the phase-two design plan for a typed, ephemeral `ToolAgentTurn` handoff that cannot leak live tool objects or retrieval evidence into persisted chat state.
- Hardened that plan with explicit SSE stream-exclusion and interrupted-run reconstruction requirements after LangGraph state-behavior review.
- Recorded ADR-0001: durable `ToolAgentTurn` reconstruction requires a saver-owned backend store; private LangGraph state cannot meet the streaming and resume constraints in the current topology.
- Revised the SQLite-backed `ToolAgentTurnRecipeStore` design after Agent Server lifecycle review: durable user-turn keys replace unstable run-only lookup, leases use renewal and fencing, terminal cleanup waits for a durable checkpoint, and the specification covers idempotent setup, rollback, pruning, thread copy, orphan reconciliation, configuration drift, and at-least-once tool execution.
- Added the saver-owned durable recipe-store foundation with canonical immutable recipes, shared-connection SQLite schema, renewable fenced leases, origin-run/thread cleanup primitives, transactional Agent Server run rollback cleanup, continuation-link reachability protection, and no graph-node or stream-contract integration yet.
- Reconstructed MCP and mixed tool turns from saver-owned immutable recipes at setup, tool-loop, and composition boundaries; added checkpointed interrupt/resume coverage, explicit MCP configuration-drift rejection, and removed the legacy mutable runtime-context turn handoff.
- Restored mixed-mode Oracle evidence across reconstruction by reading persisted `ToolMessage` artifacts, with exception-safe composition lease release and regression coverage for citations/reranking inputs.
- Hardened Ticket 03 replay semantics: checkpointed Document artifacts normalize after the latest HumanMessage, setup remains conflict-detecting while reconstruction is load-only, recipe mode/round limits are immutable, leases renew before external calls, and mixed synthesis uses the stored model.
- Completed Ticket 04 replay fidelity: checkpointed Oracle retrieval failures retain an explicit error discriminator, implicit MCP selection records all configured servers and rejects deleted definitions, load-only reconstruction avoids mutable defaults, and additive reranker recipe fields remain backward compatible.
- Completed Ticket 05 conservative recipe lifecycle: checkpoint writes atomically record saver-owned turn reachability; startup/periodic reconciliation expires only retention-aged run provenance and logs identifier/reason pairs; thread/run/prune/copy cleanup respects retained checkpoints and active leases; terminal recipes are retained until reachability is removed because the pinned SQLite saver cannot prove post-checkpoint quiescence.
- Completed durable recipe-store validation for Tickets 04–05: 106 focused tests and 5 live Agent Server MCP/mixed tests passed; MCP SSE returned `42`, mixed SSE returned retrieval plus calculator output with 2 citations, and static `mcp_compose` interruption resumed. A local process restart against the same checkpoint database resumed composition with a stable terminal ID, cleared lease, retained checkpoint links, and local-dev ephemeral thread-catalog recreation under the same ID; this is not a production restart guarantee. Langfuse trace `aced01a5a7d6770577b0378503fd632f` preserved the expected `chat_agent -> mcp_setup -> mcp_agent -> call_llm -> calculator_basic_arithmetic -> call_llm -> mcp_compose` hierarchy with all four IO-marker scans false.
- Final audit hardening delivered a reducer-safe single stable terminal answer, off-thread reranking under lease heartbeat, a secret-free MCP compatibility digest, reusable OAuth providers, and cancellation-safe MCP client eviction; final validation was 276 passed/20 skipped plus 5 live tests, including reranker-enabled mixed live success.
- Increased the RAG-mode retrieval candidate count from five to ten so reranking and answer synthesis have a broader set of Oracle documentation excerpts.

## 2026-08-17

- Completed the self-hosted Langfuse v4 migration in `events_only` mode without a historic v3 backfill; preserved legacy tables as an archive while new traces use the v4 observations data model.

## 2026-08-05

- Fixed LangGraph MCP and mixed-mode tool loops to honor the configured `MCP_MAX_ROUNDS` limit instead of using a hard-coded ten-round cap.

## 2026-08-04

- Replaced mixed-mode Oracle retrieval's private tool-state handoff with invocation-linked, turn-scoped retrieval evidence, preserving existing reranking, citations, synthesis fallback, no-context, and retrieval-failure outcomes.

## 2026-08-03

- Deepened MCP and mixed-mode tool preparation into one typed tool-turn module, eliminating the duplicated multi-key runtime-context protocol while preserving tool execution, mode-specific Oracle retrieval, and chat outcome contracts.
- Fixed frontend retry, recovery-mode, and resume actions to fork from the latest user turn's parent LangGraph checkpoint, so a rerun replaces the failed branch instead of appending a duplicate user message to the thread.
- Fixed the Stop control to wait for LangGraph cancellation before confirming success and to surface cancellation failures.
- Removed the redundant post-completion thread-state request; finalized chat rendering now uses the native LangGraph `stream.values.messages` projection while `stream.messages` continues to supply live token and tool activity.

## 2026-07-31

- Aligned the application Langfuse integration with the self-hosted v4 stack: upgraded the Python SDK to 4.14.1, removed the legacy `LANGFUSE_ENVIRONMENT` alias, and now provide environment/release through the v4 environment-variable contract.
- Normalized Langfuse metadata to v4's string and 200-character attribute limit without truncating trace input or output.
- Updated Langfuse investigation guidance to use the v4 Observations API instead of the deprecated trace-list endpoint.

## 2026-07-02

- Bumped version to 0.2.0 (significant refactoring: MCP sub-graph architecture, LangGraph Agent Server adoption, data flow cleanup).
- Added Release GitHub Actions workflow triggered by `v*.*.*` tags.
- Configured Dependabot with pip, npm, github-actions, and docker ecosystems.
- Updated AGENTS.md with release process, frontend checks (Ultracite, knip), and removed stale `ChatRuntimeService` reference.
- Fixed stale code references in `docs/MCP-USAGE.md`: removed references to deleted `mcp_agent.py` and `get_mcp_answer_async`; updated MCP mode table, flow diagram, and implementation section to current sub-graph architecture.
- Fixed stale directory reference in `docs/README.md` Repo Map: `src/rag_agent/workflows/` replaced with `src/rag_agent/graphs/`.
- Untracked `docs/superpowers/` plans and specs; added `*` to `docs/superpowers/.gitignore` so local development plan files are kept on disk but ignored in git.

## 2026-07-01

- Aligned `.env.example` with `api/settings.py`: updated stale `MODEL_LIST` / `MODEL_DISPLAY_NAMES` examples to the current region-derived defaults, removed non-existent `ENABLE_PERSISTENT_MEMORY` and `MCP_AGENT_MODEL_TIMEOUT_SECONDS`, and added missing `RAG_RETRIEVAL_TOP_K`.
- Fixed remaining stale runtime references in docs: `docs/LOGGING-ANALYTICS.md` now points to the LangGraph Agent Server health endpoint (`localhost:2024/health`) instead of the removed standalone FastAPI backend port `3002`; `observability/langfuse/README.md` no longer says the FastAPI app sends traces to Langfuse.
- Synced `docs/README.md` with the current root `README.md` so the docs-site overview page no longer carries outdated architecture, feature descriptions, and broken relative links; fixed image references so they resolve both on GitHub (`../images/`) and in the docs site (`/images/`), and updated `docs-site/scripts/sync-docs.mjs` to copy repo-root images into `public/images/` during sync.
- Updated Grafana dashboards to use current Agent Server paths (`/threads/` instead of the removed `/api/langgraph/threads/`) and replaced the stale `rag_retrieval_fallback` signal with `rag_retrieval mode=`.
- Regenerated OpenAPI baseline to reflect the `thread_id` addition to the suggestions schema, unblocking the regression guard.
- Fixed frontend environment documentation: added `FASTAPI_BACKEND_URL` to `frontend/env.example`, documented `NEXT_PUBLIC_LANGGRAPH_API_BASE` / `LANGGRAPH_BACKEND_URL` in `docs/CONFIGURATION.md`, and corrected the Docker Compose server-side URL variable in `docs/GETTING-STARTED.md`.
- Corrected `docs/DOCUMENT-POPULATION.md` to reference the actual `CHUNK_SIZE` / `CHUNK_OVERLAP` defaults from settings instead of hard-coded outdated values.
- Removed the stale `MAX_MSGS_IN_HISTORY` reference from `README.md` and corrected the MCP `semantic_search` mode description to match the code (only `vector` is supported).
- Documented the `LOGGING_ANALYTICS_MIN_LEVEL` setting in `docs/CONFIGURATION.md`.
- Stopped OTEL export errors in the LangGraph Docker container when observability is disabled by gating OTLP log export on `ENABLE_OTEL_TRACING` in `src/rag_agent/utils/logging_config.py` and defaulting `OTEL_ENABLED=false` for the `langgraph` service in `docker-compose.yml`, since LangGraph auto-enables OTEL when `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is present.
- Fixed the MCP Docker Compose startup failure by auto-creating the shared external `mcp-net` network in `mcp/run-mcp.sh` when it is missing, matching the existing base-image bootstrap behavior.
- Removed the obsolete `src.rag_agent.core.config` compatibility facade and routed Langfuse, OTEL tracing, and OTEL logging configuration through canonical `api.settings` values and standard environment variables.
- Simplified chat memory around native LangChain messages by removing the custom dictionary conversion, legacy content repair, and unused message merge helpers.
- Removed the unused `runtime.agent.normalize_messages` request-shape helper and its obsolete export/tests.
- Removed the legacy `run_mixed_node` compatibility wrapper and migrated its tests to the active `mixed_mcp` and `mixed_compose` graph nodes.
- Removed Langfuse payload masking and attribute truncation so local traces retain complete inputs, outputs, tool schemas, and reasoning data.
- Removed unused Langfuse child-observation and generation-outcome helper abstractions.
- Bound Langfuse's standard LangChain callback handler once at the LangGraph Agent Server graph boundary instead of creating tracing roots inside mode nodes.

## 2026-06-30

- Split mixed-mode execution into visible LangGraph route, retrieval, MCP, and composition nodes; added correlation metadata, semantic MCP child spans, and explicit truncated-response outcomes to Langfuse tracing.
- Kept `chat.request` as the sole Langfuse trace name so nested MCP/tool spans no longer rename the enclosing chat trace.
- Linked suggestions traces to chat sessions and added request/outcome metadata for Langfuse debugging.
- Fixed suggestions generation for tool-backed responses whose final stream item is not the assistant message.
- Linked suggestions to the effective LangGraph stream thread for newly created conversations.
- Reduced the suggestions completion budget to improve follow-up chip latency.
- Reduced local Langfuse ClickHouse CPU and disk churn by cleaning accumulated system logs and disabling remaining system-log tables.

- Reduced backend Docker build overhead by excluding local/tooling artifacts from the build context, sharing the uv cache across concurrent builds, avoiding recursive ownership conversion for the read-only virtualenv, and removing the redundant post-source dependency sync.
- Cleaned post-migration leftovers by wiring the standalone RAG MCP server to the canonical `chat_agent` graph, deleting the unreferenced checkpoint state type, correcting current startup/observability documentation, and ignoring local frontend/tooling artifacts.
- Restored native tool-call cards when rehydrating persisted threads by reconstructing calls from stored assistant/tool messages and preferring live lifecycle data when available.
- Updated native tool-call presentation to follow the AI Elements Tool pattern with a real collapsible header/content structure, state-aware status badges, compact running calls, and automatic expansion for completed or errored calls.
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

## 2026-07-01

- Fixed RAG-mode assistant message identity so streamed and replayed answers render once.
- Restored status-based frontend stream reconciliation so finalized RAG state replaces, rather than joins, live assistant output.
- Restored post-run authoritative thread-state hydration so the final RAG message uses the persisted graph message identity and cannot coexist with the live model projection.
- Added the initial LangGraph Agent Server bootstrap surface with `langgraph.json` and a minimal `chat_agent` graph while the legacy chat surface still coexists during the compatibility phase.
- Fixed LangGraph direct/RAG routing to use runtime context instead of graph state, reject unsupported `mcp`/`mixed` modes explicitly, move blocking RAG retrieval work off the Agent Server event loop, and cover the graph-mode contract with deterministic workflow tests plus real SDK integration tests.
- Added LangGraph graph-owned `mcp` and `mixed` routes, including live MCP/calculator and mixed retrieval-plus-tool verification against the Agent Server.
- Fixed LangGraph MCP config resolution to load the UI-managed MCP server store from the repo root during Agent Server runs.
- Added a first-class `langgraph` Docker Compose service, updated the core Docker stack to start backend + LangGraph + frontend together, and pointed the containerized frontend at the internal `http://langgraph:2024` service by default.
- Pinned the local Agent Server runtime to `langgraph-api>=0.10,<0.11` so the Docker LangGraph service matches the `@langchain/react` event-streaming protocol used by the frontend.
- Fixed the LangGraph chat graph to use standard Agent Server message state: graph nodes now emit `AIMessage` outputs with attached reference metadata, LangChain content blocks stay structured instead of being stringified, and persistence remains server-owned per LangGraph Agent Server conventions.
- Fixed the frontend LangGraph submit path to pass mode, collection, model, tracing, and reranker settings as top-level run `context` instead of embedding them inside graph input state, so live Agent Server runs route to the selected direct/RAG/MCP/mixed mode.

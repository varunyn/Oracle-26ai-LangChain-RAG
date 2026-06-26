# LangGraph Agent Server Adoption Design

Date: 2026-06-26

## Goal

Move the chat runtime to full LangGraph Agent Server adoption so the app no longer owns custom LangGraph protocol compatibility code. The end state should use standard LangGraph threads, runs, state, history, and streaming while preserving the product functionality the current UI already provides: model selection, RAG collections, direct/RAG/MCP/mixed modes, document ingestion, citations, feedback, tracing, and MCP settings.

The migration should remove custom code where LangGraph already provides the behavior. It should not remove product-specific behavior merely because the current implementation is custom.

## Chosen Approach

Use full backend adoption with the current product UI preserved.

LangGraph Agent Server becomes the primary chat/thread/run/stream server. The frontend keeps the current app shell and workflow controls, but its streaming layer moves toward the standard `useStream` pattern used by LangGraph and `agent-chat-ui`. FastAPI remains for non-chat product APIs unless a later rewrite chooses to move those surfaces too.

This approach avoids a wholesale frontend rebuild while still eliminating the largest source of avoidable custom code: `api/routes/langgraph_server.py` and the runtime methods that make `ChatRuntimeService` act like a graph server.

## Architecture

The target backend has three clear surfaces:

- A real LangGraph graph exported from the repo and registered in `langgraph.json`.
- LangGraph Agent Server owning chat protocol concerns: threads, runs, state, history, checkpointer-backed memory, streaming, and lifecycle events.
- FastAPI owning product APIs that are not core LangGraph protocol.

The likely graph package layout is:

- `src/rag_agent/graphs/chat_agent.py`: exported compiled graph for `langgraph.json`.
- `src/rag_agent/graphs/state.py`: typed state and context schema.
- `src/rag_agent/graphs/nodes/direct.py`: direct model answer node.
- `src/rag_agent/graphs/nodes/rag.py`: Oracle retrieval and RAG synthesis nodes.
- `src/rag_agent/graphs/nodes/mcp.py`: MCP tool agent node.
- `src/rag_agent/graphs/nodes/mixed.py`: mixed retrieval plus tool orchestration.
- `src/rag_agent/graphs/nodes/references.py`: citation, trace, reranker, usage, and MCP metadata assembly.
- `src/rag_agent/graphs/tools/`: graph-facing retrieval and MCP tool integration when tool form is appropriate.

The graph state should stay close to standard LangGraph chat state:

```python
{
    "messages": [...],
    "context": {
        "model_id": "...",
        "collection_name": "...",
        "mode": "direct|rag|mcp|mixed",
        "enable_reranker": True,
        "enable_tracing": True,
        "mcp_server_keys": [...],
    },
    "references": {...},
}
```

FastAPI can remain available through its existing app or through `langgraph.json` `http.app`, but it must not shadow built-in LangGraph routes such as `/threads` or `/runs`.

## FastAPI Boundary

FastAPI should keep product endpoints that do not duplicate LangGraph Agent Server:

- `/health`
- `/api/config`
- `/api/documents/*`
- `/api/suggestions`
- `/api/feedback`
- `/api/mcp/*`

The following custom compatibility endpoints are migration targets for deletion:

- `/api/langgraph/threads`
- `/api/langgraph/threads/{thread_id}/commands`
- `/api/langgraph/threads/{thread_id}/stream/events`
- adapter-only state and history shims under `/api/langgraph/*`

`ChatRuntimeService` should stop owning thread hydration, thread state snapshots, homemade event buffering, `astream_events`, and LangGraph protocol conversion. Its remaining reusable logic can be kept temporarily as helpers while graph nodes are introduced.

## Frontend Shape

The current product UI should be preserved unless an upstream pattern clearly reduces custom code without dropping functionality.

The stream layer should move toward a central provider similar in spirit to `agent-chat-ui`:

- `LangGraphStreamProvider` owns `apiUrl`, `assistantId`, `threadId`, and `onThreadId`.
- `assistantId` points to the graph id from `langgraph.json`, for example `chat_agent`.
- Chat submit sends standard LangGraph input: `{ messages, context }`.
- Stream options request standard channels needed by the UI, likely `["values", "tools"]`.
- Thread IDs come from `useStream` `onThreadId`.
- Thread search/history should come from the LangGraph client instead of being reconstructed primarily from local storage.
- Message rendering should prefer `stream.messages`, standard tool messages, `stream.toolProgress`, and `stream.values` before custom projection helpers.

Custom frontend code remains justified for product-specific UI:

- collection and model selectors
- direct/RAG/MCP/mixed mode controls
- document upload and indexing progress
- citations, source strips, reranker docs, and context usage display
- feedback submission with trace IDs
- MCP settings
- recovery actions such as retry, direct fallback, and RAG-only fallback

## Data Flow

User submit should follow this shape:

```ts
stream.submit(
  {
    messages: [{ role: "user", content: text }],
    context: {
      model_id,
      collection_name,
      mode,
      enable_reranker,
      enable_tracing,
      mcp_server_keys,
    },
  },
  {
    streamMode: ["values", "tools"],
  },
);
```

The graph reads context from state or config, executes the selected path, and appends standard messages, tool progress, and product references to graph state. The frontend renders:

- `stream.messages` for conversation messages.
- standard tool progress/tool messages for tool activity.
- `stream.values.references` or assistant metadata for citations, trace ID, reranker docs, model usage, and context usage.

The rule is that product settings are graph context, not route-level protocol extensions duplicated across `metadata`, `context`, and `configurable`.

## Error Handling

LangGraph-native run errors should handle unexpected failures. Expected product failures should become structured graph output:

- no Oracle context
- retrieval unavailable
- MCP tool failure
- model/provider failure that can be safely summarized for the user

Frontend recovery actions remain product features, but they resubmit standard graph input:

- Retry resubmits the last user message.
- Recover direct resubmits with `mode: "direct"`.
- Recover RAG only resubmits with `mode: "rag"`.

`stream.error` should drive unexpected run-level failure UI. Product fallback messages should render as assistant output with references/error metadata where useful.

## Migration Phases

### Phase 1: Establish Real LangGraph Server

Add `langgraph.json` and a minimal exported graph that runs through `uv run langgraph dev`. The graph may initially wrap existing runtime helpers, but it must be callable through the real Agent Server API with a simple `{ messages }` input.

Acceptance:

- `uv run langgraph dev` starts the graph server.
- A direct chat run works through the LangGraph API.
- The frontend can be configured to point at the real Agent Server in a development path.

### Phase 2: Move Runtime Semantics Into Graph Units

Move direct, RAG, MCP, and mixed behavior into graph nodes and graph tools. Decompose `ChatRuntimeService` into reusable helpers where needed, then delete runtime-owned protocol behavior.

Acceptance:

- Direct, RAG, MCP, and mixed paths run as graph paths.
- Thread memory comes from LangGraph state/checkpointing, not custom thread hydration.
- Product references are produced by graph nodes.

### Phase 3: Rewire Frontend to Standard `useStream`

Replace the current stream adapter with a LangGraph stream provider and standard submit payload. Keep the product UI and workflow controls.

Acceptance:

- Chat submit, retry, stop, and recovery actions use standard `useStream`.
- Thread id persistence uses `onThreadId`.
- Thread history comes from LangGraph server APIs.
- The UI no longer depends on the custom `/api/langgraph` adapter.

### Phase 4: Replace Custom Tool and Citation Projection

Use standard stream outputs where they fit, and keep only product-specific rendering.

Acceptance:

- Tool activity prefers standard tool progress/tool messages.
- Citations and reranker docs are read from graph state or message metadata.
- Adapter-specific projection helpers are deleted or reduced to product-only helpers.

### Phase 5: Delete Adapter and Dead Tests

Remove `api/routes/langgraph_server.py`, protocol-event buffering, adapter-only state/history shims, and tests that validate the homemade protocol rather than product behavior.

Acceptance:

- No frontend code calls `/api/langgraph/*`.
- API docs and Bruno collections no longer advertise the custom adapter endpoints.
- Tests cover product behavior through real LangGraph server paths.

## Testing Strategy

Deterministic tests are useful as guardrails, but they are not enough for this migration.

Keep small deterministic tests for pure helpers:

- state/context normalization
- citation formatting
- reference payload assembly
- request/config validation
- route boundaries for remaining FastAPI product APIs

For direct, RAG, MCP, and mixed graph paths, acceptance requires real configured calls:

- direct chat using the configured model/provider
- RAG retrieval plus answer synthesis using configured retrieval resources
- MCP tool call using configured MCP settings/tools
- mixed MCP plus retrieval behavior
- frontend stream rendering against the real LangGraph Agent Server path

These should be gated integration or e2e tests. If the provider, database, MCP server, or credentials are unavailable, the test must skip with an explicit reason. It must not silently fall back to fake behavior and claim parity.

Fake model/tool workflow tests may remain as fast regression guards, but they are secondary. The migration is not complete until the live configured Agent Server path is exercised.

## Deletion Targets

Primary deletion targets:

- `api/routes/langgraph_server.py`
- protocol event buffer code
- custom thread command/event/state/history endpoint tests
- frontend helpers that exist only to adapt homemade protocol output
- docs and Bruno examples for `/api/langgraph/*`

Conditional deletion targets after graph migration:

- `ChatRuntimeService.astream_events`
- `ChatRuntimeService.get_state`
- custom thread state store paths that duplicate LangGraph checkpoint state
- manual message hydration logic that LangGraph state replaces

## Non-Goals

This migration should not initially:

- remove document ingestion
- remove MCP settings
- remove feedback
- remove citations/source strips
- redesign the whole frontend visual layout
- move every product API out of FastAPI
- require deployed LangSmith/LangGraph production infrastructure before local development works

## Open Implementation Decisions

These should be resolved during implementation planning:

- whether `context` should be stored as graph state, run config, or both for best LangGraph compatibility
- whether FastAPI runs separately in Docker or is mounted through `langgraph.json` `http.app`
- how to represent citations: `stream.values.references`, assistant message metadata, custom UI events, or a combination
- whether MCP tools should be graph tools directly or wrapped through the existing MCP agent helper during the first phase
- how production Docker Compose should run the frontend, LangGraph Agent Server, FastAPI product APIs, vector database, and observability services

## Rollout Risks

Main risks:

- RAG and MCP behavior may differ once thread state and tool calls are owned by LangGraph instead of custom runtime code.
- Live model/tool behavior is nondeterministic, so acceptance checks must validate capabilities and output shape rather than exact prose.
- The current UI has product-specific behavior that upstream `agent-chat-ui` does not include; replacing too much frontend at once would risk losing working product flows.
- Running FastAPI through `http.app` could accidentally shadow LangGraph routes if path boundaries are not enforced.
- Local Docker and docs need to change together, or developers will keep starting the old FastAPI chat path.

## Success Criteria

The migration is successful when:

- Chat execution runs through real LangGraph Agent Server.
- The frontend uses standard `useStream` against the graph server.
- Direct, RAG, MCP, and mixed modes pass gated live integration/e2e checks using configured providers and tools.
- Product UI behavior remains available: models, collections, modes, uploads, citations, feedback, tracing, and MCP settings.
- The custom `/api/langgraph` compatibility router is deleted.
- Docs, tests, and local commands describe the Agent Server path as the default.

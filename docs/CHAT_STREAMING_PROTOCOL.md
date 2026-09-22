# Chat streaming protocol

The browser uses the thread-centric protocol provided by `@langchain/react`:

- `POST /threads/{thread_id}/commands` to submit runs and control commands
- `POST /threads/{thread_id}/stream/events` for the persistent event stream

## Streaming

The frontend creates one `@langchain/react` `StreamProvider` at the chat page
boundary, with `assistantId: "chat_agent"`. Child components observe that same
stream through `useStreamContext`; they do not create another chat store or
transport. It submits standard `messages` input plus top-level runtime
`context` and receives Agent Server streaming updates.

Application-level run input:

```json
{
  "assistant_id": "chat_agent",
  "input": {
    "messages": [{ "role": "user", "content": "Hello" }]
  },
  "context": {
    "mode": "rag"
  },
  "stream_mode": ["values", "messages", "tools", "lifecycle"]
}
```

Notes:

- A terminal root `lifecycle` event settles the active run. The persistent
  event transport remains open; an unexpected close is a connection failure
  and activates the bounded reconnect policy.
- Assistant references and sources are carried in `response_metadata` / `additional_kwargs` on assistant messages.
- Frontend uses `@langchain/react` directly against `NEXT_PUBLIC_LANGGRAPH_API_BASE` (local default: `http://localhost:2024`).
- FastAPI no longer adapts or proxies the chat protocol.

## Frontend state ownership

`@langchain/react` is the single owner of transport state, streamed messages,
tool calls, thread hydration, checkpoint metadata, loading state, and root
stream errors. The application-owned layer only maps those projections to
product presentation and supplies mode/context policy.

| Concern | Authoritative source | Application use |
| --- | --- | --- |
| Live and replayed message timeline | `stream.messages` | Ordered message presentation, citations, and context-usage extraction |
| Live tool activity | `stream.toolCalls` | Running, completed, and failed tool cards |
| Replayed tool activity | Persisted `AIMessage.tool_calls` + matching `ToolMessage` results | Checkpoint-derived tool cards when the SDK root projection hydrates empty |
| Application progress | `stream.values.progress` | Progress indicator; never a second message history |
| Thread hydration | `stream.isThreadLoading` | `hydrating` lifecycle state while a selected thread loads |
| Active run | `stream.isLoading` | `running` lifecycle state and stop affordance |
| Terminal stream failure | `stream.error` | `failed` state and one recovery action |
| Reconnect attempt | `onReconnect` / `onConnected` callbacks | `reconnecting` feedback with attempt, cause, and delay |
| Retry/fork origin | `useMessageMetadata(...).parentCheckpointId` | Checkpoint-aware retry and regeneration |
| Thread history | Agent Server `threads.search(...)` | Sidebar; browser storage keeps only the selected ID |

The product lifecycle is `hydrating`, `idle`, `running`, `reconnecting`, or
`failed`. It is derived only from the documented stream state and connection
callbacks. A local submit, delete, or history-refresh error remains scoped to
that operation and does not become a second transport lifecycle.

The native tools projection is populated from the Agent Server `tools` channel.
Each assembled call has a stable `callId`, `name`, `input`, `output`, `status`,
and optional `error`. The UI maps those documented fields directly. It does
not parse tool calls out of message content or accept legacy aliases.

During idle root-thread hydration, the current SDK does not seed
`stream.toolCalls` from checkpoint messages. The integration boundary therefore
hydrates the same typed call lifecycle from standard `AIMessage.tool_calls` and
matching `ToolMessage.tool_call_id` records. Native live lifecycle events replace
that checkpoint seed by `callId`; no tool names, content blocks, or legacy fields
are inferred.

The SDK owns command envelopes, event sequence bookkeeping, reconnect, and
subscription state. Application code calls `stream.submit(...)` rather than
constructing those wire messages itself. For protocol debugging, inspect the
browser requests to the two endpoints above and the root `values`, `messages`,
`tools`, and `lifecycle` events returned by the server.

## Server-owned memory: delta-only input + thread IDs

- The LangGraph Agent Server is the source of truth for conversation context and thread state.
- API requests should contain at least one user message in `input.messages`.
- `thread_id` is the conversation identifier.
  - Frontend persists only the active `thread_id` as a reopen pointer and reuses it on later turns.
  - Sidebar history is loaded from the Agent Server `threads.search(...)` response and is not persisted in browser storage.

### Tool activity channels

- Native Agent Server tool calls use the `tools` channel and are projected by `@langchain/react` as `stream.toolCalls`.
- MCP history/replay metadata is retained on the final assistant message in `mcp_tool_invocations`.
- The graph emits normal LangChain `AIMessage`, `ToolMessage`, and final
  `AIMessage` updates in its `messages` state. The Agent Server therefore owns
  the persisted transcript and can replay it through the same native message
  projection after hydration.

### Inspecting + deleting thread state

- Inspect through the Agent Server:

```bash
curl -s http://localhost:2024/threads/search
curl -s http://localhost:2024/threads/<thread_id>/state
```

- Delete via LangGraph client or Agent Server thread APIs.
- Reset all state: delete the thread through Agent Server APIs or clear the configured checkpoint storage.

## Verification boundary

Deterministic browser fixtures cover the frontend protocol contract, including
message ordering, native tool settlement, replay, retry, thread switching,
deletion, reconnect, and terminal failure. Those fixtures prove UI behavior
against the documented event shapes only. A live Agent Server run is still
required to prove provider behavior, persisted replay, real tool output, and
network recovery; unavailable infrastructure must be reported as an explicit
proof gap rather than inferred from the fixtures.

# Chat Streaming Controller Refactor Design

Date: 2026-06-25

## Goal

Refactor the frontend chat streaming controller so the code is easier to understand, review, and maintain without changing user-visible behavior or backend contracts.

The first implementation targets only `frontend/src/hooks/chat/useChatController.ts`. It must preserve the current FastAPI/LangGraph integration, recent LangGraph v1 stream behavior, duplicate submitted prompt fix, RAG references, MCP progress display, suggestions, feedback, and recovery actions.

## Context

The frontend currently has 6,661 TypeScript/TSX lines under `frontend/src`. The largest file is `frontend/src/hooks/chat/useChatController.ts` at 956 lines. That file owns several concerns at once:

- `@langchain/react` `useStream` configuration and lifecycle.
- LangGraph submit payload construction.
- Conversion of `stream.messages` and `stream.values.messages` into UI messages.
- Reference payload extraction from LangChain metadata and raw stream values.
- MCP tool-call progress conversion.
- Pending user-message reconciliation for optimistic/streamed output.
- Stream debug flags and logging.
- Error toasts.
- Feedback submission.
- retry, recover, resume, stop, clear, and suggestions actions.

The LangGraph `useStream` documentation says the hook should be mounted once per thread and owns thread lifecycle plus projections such as `values`, `messages`, and `toolCalls`. The current app is broadly aligned because `ChatPageContent` is keyed by `threadId`, and the controller binds `assistantId`, `apiUrl`, and `threadId`. The maintenance issue is that projection, fallback, debug, and UI action logic are all concentrated in one hook.

## Non-Goals

- Do not rewrite the app around the LangChain deployment-cookbook `js-next` sample.
- Do not move chat execution into Next route handlers.
- Do not change FastAPI routes, request payload contracts, or stream response contracts.
- Do not split `frontend/src/app/settings/page.tsx` in this implementation.
- Do not add a new frontend unit-test runner in this implementation.
- Do not redesign chat UI components.

## Chosen Approach

Use a behavior-preserving stream-boundary refactor.

Keep `useChatController(args)` as the public hook consumed by `frontend/src/app/page.tsx`. Split its internal logic into focused modules under `frontend/src/hooks/chat/` and `frontend/src/lib/chat/`, while keeping returned controller fields and component props unchanged.

This keeps the first implementation reviewable and avoids mixing structural cleanup with backend or UI behavior changes.

## Proposed Module Boundaries

### `useChatController.ts`

Owns high-level orchestration only:

- local controller state that is directly UI-facing.
- one `useStream` instance for the active thread.
- calls into message projection helpers.
- wires suggestions and scroll behavior.
- returns the same controller shape currently used by the page.

It should no longer contain low-level parsing, stream debug summarization, or tool-call conversion helpers.

### `stream-config.ts`

Owns LangGraph stream configuration helpers:

- resolve the LangGraph API URL from frontend API base configuration.
- build the submit payload from body params and an optional mode override.
- keep the current mirrored `context`, `metadata`, and `configurable` fields.

This module must preserve the current submitted payload shape:

- `messages`
- `model`
- `session_id`
- `collection_name`
- `enable_reranker`
- `enable_tracing`
- `mode`
- `context`
- `metadata`
- `configurable`

### `message-projection.ts`

Owns conversion from stream state to UI messages:

- map `stream.messages` into `MessageLike[]`.
- map `stream.values.messages` into `MessageLike[]`.
- choose the current preferred source of messages without changing behavior.
- merge pending user messages only when the streamed/server-projected messages do not already contain that submitted text.
- attach fallback references to the latest assistant message when class-message metadata drops reference payloads.
- attach live MCP progress events.

The duplicate submitted prompt behavior is part of this module's contract.

### `references.ts`

Owns reference extraction and normalization:

- extract `MessageReferences` from LangChain `additional_kwargs`.
- extract `MessageReferences` from raw `values.messages`.
- normalize context usage fields.
- expose small, typed helpers used by projection and controller state effects.

Citation, reranker, MCP progress, trace id, and error fields must stay compatible with existing chat components.

### `tool-progress.ts`

Owns conversion from `AssembledToolCall[]` to `McpProgressEvent[]`.

This keeps MCP display behavior isolated from message projection and makes it clear which parts come from live SDK tool-call assembly versus backend reference payloads.

### `stream-debug.ts`

Owns debug-mode behavior:

- query-param and localStorage debug flags.
- break-on-event behavior.
- stream status, message, value, and tool progress summaries.
- guarded console logging.

No production behavior should depend on this module.

### `useChatActions.ts`

Owns UI actions that depend on controller state and stream methods:

- submit
- retry
- recover as direct
- recover as RAG
- resume
- stop
- clear chat
- feedback submission

This hook should receive already-projected messages and callbacks instead of reaching into projection internals.

## Data Flow

1. `frontend/src/app/page.tsx` calls `useChatController(...)` with model, thread, session, collection, reranker, tracing, flow mode, toast, and clear-chat dependencies.
2. `useChatController` computes body params with `useChatBodyParams`.
3. `useChatController` mounts one `useStream` instance for the active thread.
4. `useChatActions` submits the same LangGraph payload shape currently used by the app.
5. `message-projection.ts` combines `stream.messages`, `stream.values.messages`, pending user message state, and `stream.toolCalls` into the `messages` array consumed by `ChatMessageList`.
6. Controller effects derive context usage and search-error toasts from the last assistant message references.
7. `useSuggestions` continues to run from projected messages and status.
8. Components receive the same props as before.

## Error Handling

The refactor must preserve existing error behavior:

- Stream errors are logged and shown with a toast once per thread/error message.
- Reference payload errors still show the "Search unavailable" toast.
- Feedback submission failures still show a feedback failure toast.
- Clear chat still attempts backend thread cleanup, stops the stream when possible, clears local state, and shows a success toast.
- Stop still calls `stream.stop()` and shows "Generation stopped".

## `useStream` Best-Practice Alignment

The implementation should keep these current aligned behaviors:

- Mount `useStream` once per active thread.
- Keep `threadId` binding stable for the active chat.
- Use `stream.stop()` for cancellation.
- Treat `stream.messages`, `stream.values`, and `stream.toolCalls` as the primary stream projections.

The implementation should improve maintainability around these points:

- Make the manual projection layer explicit instead of burying it inside the controller hook.
- Keep pending-user-message reconciliation isolated and documented by code structure.
- Consider `onCompleted` only if it can simplify settled-run effects without behavior drift. If it changes timing or risks duplicate suggestion fetches, leave it for a later pass.
- Do not change `optimistic` behavior in this refactor. Evaluating `optimistic: false` is a separate behavior change because the current duplicate-prompt fix depends on explicit pending-message handling.

## Verification

Run these checks after implementation:

```bash
pnpm --dir frontend build
pnpm --dir frontend test:e2e
```

The existing e2e suite should protect these behaviors:

- Generic suggestions render on first load.
- Clicking a suggestion does not render duplicate user messages.
- Mocked LangGraph protocol `values` events render assistant responses.
- Source/reference rendering remains visible.
- Thread history and active thread switching remain intact.

If the full e2e suite cannot run in the current environment, report the exact failure and run the most targeted available subset.

## Rollout Plan

Implement in small mechanical steps:

1. Extract shared types needed by the controller and helper modules.
2. Extract pure reference and context-usage helpers.
3. Extract tool-progress conversion.
4. Extract message projection.
5. Extract stream debug helpers.
6. Extract action callbacks.
7. Reduce `useChatController.ts` to orchestration.
8. Run frontend build and e2e checks.

Each step should avoid changing component props or backend payload shape.

## Success Criteria

- `useChatController.ts` is substantially smaller and mostly orchestration-focused.
- Extracted modules have clear responsibility boundaries.
- No chat UI behavior changes are intentional in this pass.
- Frontend build passes.
- Existing chat streaming e2e tests pass or any environment-specific failures are documented with concrete logs.

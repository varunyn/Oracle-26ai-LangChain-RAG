# Frontend Stream Simplification Design

## Goal

Reduce duplicate frontend chat-state handling while preserving the current LangGraph Agent Server behavior: optimistic message IDs, durable threads, structured content, citations, retries and recovery, replay deduplication, and visible MCP tool progress. The resulting path must contain no legacy transcript formats or compatibility fallback branches.

## Context

The frontend correctly uses `useStream` from `@langchain/react`. `@langchain/core` is used for LangChain message types and runtime message identification; these are complementary dependencies, not competing chat frameworks.

The main duplication is that the controller currently considers both `stream.messages` and `stream.values.messages` as transcript sources, then applies projection and compatibility parsing before rendering. The simplification will establish one canonical transcript source while retaining app-specific enrichment.

## Documentation evidence

The current official LangGraph frontend overview documents `useStream` as the frontend stream handle, with `stream.messages` for streamed messages and `stream.values` for full graph state such as named state keys. The official chat examples render the message list directly from `stream.messages`; they do not prescribe assembling a second transcript from `stream.values.messages`.

The official frontend examples also use `AIMessage` and `HumanMessage` instance checks when message-specific rendering is needed. The docs commonly import those classes from the `langchain` package, while the official testing documentation demonstrates the lower-level `@langchain/core/messages` import. Therefore this design does not assume that `@langchain/core` is wrong; it requires verifying whether the current direct production imports are still necessary after the transcript path is simplified.

## Proposed architecture

```text
useStream()
  ├── stream.messages  -> canonical visible transcript input
  ├── stream.toolCalls -> MCP/tool progress projection
  ├── stream.values    -> non-message graph state only
  └── stream.submit()  -> submit, retry, recovery, and resume
```

`LangGraphStreamProvider` remains the single owner of the `useStream` instance. `useChatController` remains an orchestration hook, and `useChatActions` remains responsible for user-triggered stream operations. Message projection becomes a small adapter for application-specific fields rather than a second message store.

## Scope

### In scope

- Make `stream.messages` the sole transcript source after contract tests establish the required shape for this repository.
- Restrict `stream.values` usage to non-message graph state.
- Remove redundant message-source fallback and generic reconstruction where structured v1 messages already provide the data.
- Remove legacy stringified-content parsing, legacy `parts` parsing, raw state-message projection, and unknown-shape fallback branches from the active chat path.
- Preserve citation/reference extraction, MCP tool-progress rendering, replay deduplication, stable IDs, retry/recovery, and thread lifecycle behavior.
- Remove imports and types made unused by the simplification.
- Add or strengthen focused frontend tests and run frontend lint, unit tests, build, and relevant E2E coverage.

### Out of scope

- Changing the LangGraph Agent Server protocol or backend graph.
- Replacing `useStream` with another chat transport or UI library.
- Redesigning the UI.
- Removing `@langchain/core` unless no direct production usage remains after simplification.
- Removing durable-thread support, citations, MCP progress, or recovery behavior.

## Data-flow rules

1. The rendered transcript is derived from `stream.messages`, matching the official frontend examples.
2. `stream.values` may provide graph state such as context usage only when that state is not already represented by a message. Because the official overview describes `stream.values` as the full graph state, the local `stream.values.messages` fallback is removed only after a repository-level stream contract test proves it is redundant.
3. Tool-call progress is derived from `stream.toolCalls`; it is not synthesized from a second message transcript.
4. Message IDs from the client and Agent Server remain stable through optimistic display, streaming, replay, retry, and recovery.
5. Legacy content parsing is removed; current Agent Server/LangChain structured message content is the only supported chat format.

## Testing and verification

Before simplifying, tests must cover:

- user/assistant ordering during streaming;
- optimistic client ID preservation;
- replay and duplicate-message suppression;
- structured content and citations;
- MCP/tool progress;
- retry, resume, and recovery;
- clear-chat and durable-thread reload behavior.

After each cleanup slice, run the focused Vitest tests. At the end, run `pnpm lint`, `pnpm test`, `pnpm build`, and the relevant Playwright test path. Any live Agent Server verification must use the configured frontend and backend surfaces and must report unavailable runtime dependencies explicitly.

## Success criteria

- There is one documented canonical message source in the frontend.
- No production code reconstructs a visible transcript from both `stream.messages` and `stream.values.messages`.
- No production chat path parses legacy stringified message blocks, accepts legacy `parts`, or falls back to raw state-message shapes.
- Application-specific citations, tool progress, IDs, retries, recovery, and thread behavior remain unchanged.
- `@langchain/react` remains the stream integration and `@langchain/core` remains only where its message model is directly needed.
- Focused tests and required frontend checks pass.

## Implementation notes

- The final frontend projection path is `useChatController -> projectStreamMessages -> ChatMessageList`, with `stream.toolCalls` merged in only as live MCP progress metadata.
- `stream.values` remains available on the stream object for non-message graph state, but the visible transcript no longer reads `stream.values.messages`.
- Content extraction is now intentionally strict: plain strings and structured text blocks are supported; legacy stringified arrays, `parts`, and raw unknown message shapes are not.

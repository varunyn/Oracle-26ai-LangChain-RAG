# Frontend Stream Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `stream.messages` the only frontend chat transcript source and remove legacy message formats and compatibility fallbacks without changing user-visible chat behavior.

**Architecture:** Keep one `useStream` instance in `LangGraphStreamProvider`. Pass its structured messages into a narrow projection that adds only application-specific references and MCP progress. Keep `stream.values` for non-message graph state, and keep `useChatController`/`useChatActions` for orchestration and user actions.

**Tech Stack:** Next.js 16, React 19, TypeScript, `@langchain/react` v1, `@langchain/core` message types, Vitest, Playwright, pnpm.

## Global Constraints

- `stream.messages` is the only supported chat transcript source.
- Remove legacy stringified-content, `parts`, raw-state-message, and alternate-transport fallbacks.
- Preserve optimistic IDs, durable threads, structured content, citations, replay deduplication, retries, recovery, and MCP progress.
- Keep `@langchain/react` for `useStream`; keep `@langchain/core` only where direct message classes/types remain necessary.
- Use pnpm for all frontend commands.
- Update `CHANGELOG.md` under the current date.

---

### Task 1: Lock the structured stream contract

**Files:**
- Modify: `frontend/src/hooks/chat/__tests__/message-projection.test.ts`
- Modify: `frontend/src/lib/chat/__tests__/messages.test.ts`
- Create: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`

**Interfaces:**
- Consumes: `HumanMessage` and `AIMessage` values returned by `useStream().messages`.
- Produces: tests defining the only supported message shape and retained behavior.

- [ ] **Step 1: Add canonical fixtures**

Use `HumanMessage` and `AIMessage` from `@langchain/core/messages`, with explicit IDs, string content, structured text-block content, and reference metadata.

```ts
const user = new HumanMessage({ id: "user-1", content: "What is Oracle 26ai?" });
const assistant = new AIMessage({
  id: "assistant-1",
  content: [{ type: "text", text: "Oracle 26ai is..." }],
  additional_kwargs: { citations: [{ source: "guide.pdf", page: "2" }] },
});
```

- [ ] **Step 2: Test retained behavior**

Cover ordering, stable IDs, latest-message-wins deduplication, structured text extraction, citations, and MCP progress. Do not add tests for `stream.values.messages`, `parts`, stringified Python representations, or unknown raw message shapes.

- [ ] **Step 3: Run the baseline**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/message-projection.test.ts src/hooks/chat/__tests__/stream-message-contract.test.ts src/lib/chat/__tests__/messages.test.ts
```

Expected: existing tests pass and the new contract tests establish the behavior required by later cleanup.

- [ ] **Step 4: Commit the test baseline**

```bash
git add frontend/src/hooks/chat/__tests__/message-projection.test.ts frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts frontend/src/lib/chat/__tests__/messages.test.ts
git commit -m "test: define canonical frontend stream message contract"
```

### Task 2: Remove the second transcript source

**Files:**
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Modify: `frontend/src/hooks/chat/message-projection.ts`
- Modify: `frontend/src/hooks/chat/__tests__/message-projection.test.ts`

**Interfaces:**
- Consumes: `stream.messages) and `stream.toolCalls`.
- Produces: `projectStreamMessages({ streamMessages, liveToolProgressEvents })` as the only visible-message projection entry point.

- [ ] **Step 1: Change the controller input**

Replace the current `stateMessages` extraction and `projectVisibleMessages` call with:

```ts
const liveToolProgressEvents = useMemo(
  () => toolCallsToMcpEvents(stream.toolCalls ?? EMPTY_TOOL_CALLS),
  [stream.toolCalls],
);
const messages = useMemo(
  () =>
    projectStreamMessages({
      streamMessages: stream.messages,
      liveToolProgressEvents,
    }),
  [liveToolProgressEvents, stream.messages],
);
```

Remove the `stream.values.messages` cast and any import used only by that cast.

- [ ] **Step 2: Delete state projection**

Remove `rawMessageId`, `rawMessageRole`, `rawMessageContent`, `projectStateMessages`, and `projectVisibleMessages`. Keep explicit replay deduplication, stream projection, references, and MCP progress.

- [ ] **Step 3: Update tests**

Delete state-source precedence tests. Keep stream ordering, ID deduplication, references, and MCP progress tests.

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/message-projection.test.ts src/hooks/chat/__tests__/stream-message-contract.test.ts
pnpm build
```

Expected: no production reference to `projectVisibleMessages`, `projectStateMessages`, or `stream.values.messages`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/chat/useChatController.ts frontend/src/hooks/chat/message-projection.ts frontend/src/hooks/chat/__tests__/message-projection.test.ts
git commit -m "refactor: use stream messages as the sole chat transcript"
```

### Task 3: Remove legacy message-content parsing

**Files:**
- Modify: `frontend/src/lib/chat/messages.ts`
- Modify: `frontend/src/lib/chat/__tests__/messages.test.ts`
- Modify: `frontend/src/hooks/chat/references.ts`
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/hooks/useSuggestions.ts`
- Modify: `frontend/src/hooks/chat/__tests__/message-projection.test.ts`

**Interfaces:**
- Consumes: supported string or structured text-block content.
- Produces: one strict text extraction helper with no legacy `parts`, stringified-array, broad-`unknown`, or catch-and-empty path.

- [ ] **Step 1: Replace the content helper**

Use a strict supported-content type:

```ts
type SupportedContent =
  | string
  | readonly { type?: string; text?: string }[];

export function getMessageContent(message: { content?: SupportedContent }): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return "";
  return message.content
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("");
}
```

Delete `readLegacyStringifiedTextBlocks`, `parts) handling, and the try/catch that converts malformed content to an empty answer.

- [ ] **Step 2: Remove duplicate readers**

Delete `readMessageText` and the `readText` wrapper. Update projection, references, feedback, suggestions, and message rendering to consume the strict normalized content type without `Parameters<typeof getMessageContent>[0]` casts.

- [ ] **Step 3: Update tests**

Delete tests for stringified Python-style blocks and `parts). Add tests for string content, structured text blocks, empty content, and ignored non-text blocks.

- [ ] **Step 4: Verify and commit**

Run:

```bash
cd frontend && pnpm test -- src/lib/chat/__tests__/messages.test.ts src/hooks/chat/__tests__/message-projection.test.ts
git add frontend/src/lib/chat/messages.ts frontend/src/lib/chat/__tests__/messages.test.ts frontend/src/hooks/chat/references.ts frontend/src/components/chat/ChatMessageList.tsx frontend/src/hooks/useSuggestions.ts frontend/src/hooks/chat/__tests__/message-projection.test.ts
git commit -m "refactor: remove legacy frontend message parsing"
```

### Task 4: Remove remaining compatibility branches

**Files:**
- Modify: `frontend/src/hooks/chat/message-projection.ts`
- Modify: `frontend/src/hooks/chat/references.ts`
- Modify: `frontend/src/lib/chat/messages.ts`
- Modify: `frontend/src/hooks/chat/controller-types.ts` only if types become unused
- Modify: `frontend/src/hooks/useChatSession.ts` only if the obsolete ID helper is unused

**Interfaces:**
- Consumes: documented structured LangChain messages and explicit application projection.
- Produces: no unknown-status, raw-role, obsolete-thread-ID, or legacy-shape compatibility branches.

- [ ] **Step 1: Inventory before removal**

Run:

```bash
rg -n "generateThreadId|normalizeStatus|toRole|toReferencesFromRawMessage|readText|readMessageText|stateMessages|parts\\?|fallback|legacy" frontend/src frontend/tests
```

- [ ] **Step 2: Make status handling explicit**

Replace unknown-status fallback logic with the supported `useStream` loading/error/submission lifecycle used by this app. Preserve `submitted`, `streaming`, `ready), and `error` explicitly; do not introduce a new fallback status.

- [ ] **Step 3: Remove raw message-role fallback**

After Task 1 verifies the stream message instances, use the canonical `HumanMessage`, `AIMessage`, and `SystemMessage` checks and remove the raw `type` fallback from `toRole`. Remove `toReferencesFromRawMessage) if no caller remains.

- [ ] **Step 4: Remove obsolete thread-ID fallback**

If `generateThreadId` has no caller, delete it and its tests. The active thread ID must come from the persisted session/provider flow and Agent Server `onThreadId`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/message-projection.test.ts src/hooks/chat/__tests__/thread-errors.test.ts src/hooks/__tests__/useChatSession.test.ts
pnpm lint
git add frontend/src/hooks/chat/message-projection.ts frontend/src/hooks/chat/references.ts frontend/src/lib/chat/messages.ts frontend/src/hooks/chat/controller-types.ts frontend/src/hooks/useChatSession.ts
git commit -m "refactor: remove chat compatibility fallbacks"
```

Expected: removed symbols have no production callers and focused tests/lint pass.

### Task 5: Verify dependencies, browser behavior, and documentation

**Files:**
- Modify: `frontend/package.json` only if a direct dependency is unused
- Modify: `frontend/pnpm-lock.yaml` only if `package.json` changes
- Modify: `frontend/tests/e2e/chat-streaming.spec.ts`
- Modify: `frontend/tests/e2e/chat-live.spec.ts`
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-06-29-frontend-stream-simplification-design.md`

**Interfaces:**
- Consumes: the simplified stream/controller implementation.
- Produces: verified dependency ownership, browser regression coverage, and documented canonical data flow.

- [ ] **Step 1: Audit LangChain imports**

Run:

```bash
rg -n "@langchain/(react|core)|from \\"langchain\\"" frontend/src frontend/tests
```

Keep `@langchain/react` because `useStream` and `AssembledToolCall` are used. Keep `@langchain/core` only if production code still imports message classes/types. Do not add the umbrella `langchain` package merely to match documentation examples.

- [ ] **Step 2: Add browser assertions**

Extend existing E2E coverage to assert one submitted user message, one streamed assistant response, visible tool progress when emitted, and no duplicate messages after thread reload. Do not add another transport or fallback mock.

- [ ] **Step 3: Update the changelog**

Add a dated entry describing removal of duplicate transcript sources and legacy message parsing while preserving LangGraph streaming, durable threads, citations, and MCP progress.

- [ ] **Step 4: Run final checks**

Run:

```bash
cd frontend && pnpm lint
pnpm test
pnpm build
pnpm test:e2e -- tests/e2e/chat-streaming.spec.ts tests/e2e/chat-live.spec.ts
```

If Agent Server/model dependencies are unavailable, report that exact limitation instead of substituting a fallback path.

- [ ] **Step 5: Final forbidden-code audit**

Run:

```bash
rg -n "legacy|fallback|stateMessages|parts\\?|readLegacy|stringified|projectVisibleMessages|projectStateMessages|stream\\.values.*messages" frontend/src frontend/tests
git diff --check
git status --short
```

Expected: no forbidden production chat-path matches, clean whitespace validation, and only intended files changed.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml frontend/tests/e2e/chat-streaming.spec.ts frontend/tests/e2e/chat-live.spec.ts CHANGELOG.md docs/superpowers/specs/2026-06-29-frontend-stream-simplification-design.md
git commit -m "refactor: simplify frontend LangGraph stream handling"
```

## Final verification checklist

- [ ] `stream.messages` is the only transcript source.
- [ ] `stream.values` is used only for non-message graph state.
- [ ] Legacy stringified content, `parts`, raw state projection, and unknown-shape fallbacks are removed.
- [ ] `@langchain/react` remains the frontend stream integration.
- [ ] Direct `@langchain/core` usage is justified by remaining message-class/type checks.
- [ ] IDs, durable threads, citations, MCP progress, retries, recovery, and replay deduplication remain covered.
- [ ] Frontend lint, unit tests, build, and relevant E2E tests pass.

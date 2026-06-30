# Native Tool Calls with AI Elements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom MCP progress translation layer with direct `useStream().toolCalls` to AI Elements tool rendering, while preserving the existing custom message shell, citations, feedback, retries, and recovery.

**Architecture:** Keep `useStream` as the single runtime source. Preserve each message’s native tool-call IDs during projection, pass the live `AssembledToolCall[]` separately through the controller, and match each call to its owning AI message by `callId`. Render the matched calls with the existing AI Elements `Tool`, `ToolHeader`, `ToolInput`, and `ToolOutput` components.

**Tech Stack:** Next.js 16, React 19, TypeScript, `@langchain/react` v1, AI Elements source components, Vitest, Playwright, pnpm.

## Global Constraints

- Use the official `stream.messages` plus `stream.toolCalls` model.
- Match calls by native message tool-call ID and `AssembledToolCall.callId`; never attach every call to the latest assistant message.
- Remove the custom `McpProgressEvent` translation path and legacy tool notice.
- Preserve citations, durable threads, optimistic IDs, streamed text, feedback, retries, recovery, and visible tool lifecycle states.
- Keep `ChatMessageList` and `ChatMessageItem` unless a smaller targeted change is proven safe.
- Do not change backend tool-calling or thread protocols.
- Do not change scroll behavior or introduce `Conversation` in this plan.
- Update `CHANGELOG.md` under the current date.

---

### Task 1: Establish the native tool-call contract

**Files:**
- Create: `frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts`
- Modify: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`
- Modify: `frontend/src/hooks/chat/__tests__/message-projection.test.ts`

**Interfaces:**
- Consumes: `AIMessage.tool_calls` IDs and `AssembledToolCall` values.
- Produces: tests defining one-to-one message/tool matching and lifecycle rendering inputs.

- [ ] **Step 1: Add a message fixture with multiple tool-call IDs**

Use an `AIMessage` with two structured tool calls and an `AssembledToolCall[]` containing running, finished, and error entries.

```ts
const message = new AIMessage({
  id: "assistant-1",
  content: "",
  tool_calls: [
    { id: "call-1", name: "semantic_search", args: { query: "Oracle" } },
    { id: "call-2", name: "list_documents", args: {} },
  ],
});
```

- [ ] **Step 2: Test matching behavior**

Add a pure helper test that only calls whose `callId` appears in the message’s native `tool_calls` are returned. Assert that calls belonging to another assistant message are excluded and that multiple calls remain independently ordered.

- [ ] **Step 3: Test lifecycle fields**

Assert that the UI receives `name`, `callId`, `input`/`args`, `output`, `status`, and `error` unchanged. Do not convert them to `McpProgressEvent` or phase strings.

- [ ] **Step 4: Run focused tests**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/tool-call-mapping.test.ts src/hooks/chat/__tests__/stream-message-contract.test.ts src/hooks/chat/__tests__/message-projection.test.ts
```

- [ ] **Step 5: Commit the contract tests**

```bash
git add frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts frontend/src/hooks/chat/__tests__/message-projection.test.ts
git commit -m "test: define native tool call message mapping"
```

### Task 2: Preserve native tool-call IDs during message projection

**Files:**
- Modify: `frontend/src/hooks/chat/controller-types.ts`
- Modify: `frontend/src/hooks/chat/message-projection.ts`
- Modify: `frontend/src/hooks/chat/__tests__/message-projection.test.ts`

**Interfaces:**
- Consumes: `BaseMessageWithKwargs` values from `stream.messages`.
- Produces: normalized messages with `toolCallIds?: string[]` and existing content/reference fields.

- [ ] **Step 1: Add the minimal normalized field**

Add this field to the normalized message type:

```ts
toolCallIds?: string[];
```

Do not copy the entire LangChain message object into the UI model.

- [ ] **Step 2: Project native IDs**

In `projectStreamMessages`, derive IDs only from the message’s native `tool_calls` entries:

```ts
toolCallIds: message.tool_calls
  ?.map((toolCall) => toolCall.id)
  .filter((id): id is string => typeof id === "string" && id.length > 0),
```

Omit the field when no valid IDs exist.

- [ ] **Step 3: Test preservation**

Assert that projection preserves two tool-call IDs on the owning assistant message, preserves normal messages without adding an empty field, and continues to preserve references and replay deduplication.

- [ ] **Step 4: Run tests and lint**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/message-projection.test.ts src/hooks/chat/__tests__/tool-call-mapping.test.ts
pnpm lint
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/chat/controller-types.ts frontend/src/hooks/chat/message-projection.ts frontend/src/hooks/chat/__tests__/message-projection.test.ts
git commit -m "refactor: preserve native tool call ids in messages"
```

### Task 3: Pass live AssembledToolCall values to the message list

**Files:**
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Create or modify: `frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts`

**Interfaces:**
- Consumes: `stream.toolCalls` from `useStream`.
- Produces: `ChatMessageList` receives `toolCalls: AssembledToolCall[]` alongside normalized messages.

- [ ] **Step 1: Return tool calls from the controller**

Expose the existing stream tool-call array from `useChatController`:

```ts
toolCalls: stream.toolCalls ?? [],
```

Do not convert it to another event type.

- [ ] **Step 2: Thread the prop through page composition**

Pass `chat.toolCalls` from `ChatPageContent` into `ChatMessageList`. Keep citations and message references on the existing normalized message objects.

- [ ] **Step 3: Add a pure matcher**

Create a small exported helper in `ChatMessageList.tsx` or a dedicated `tool-call-mapping.ts` module:

```ts
export function toolCallsForMessage(
  toolCallIds: readonly string[] | undefined,
  toolCalls: readonly AssembledToolCall[],
): AssembledToolCall[] {
  const ids = new Set(toolCallIds ?? []);
  return toolCalls.filter((toolCall) => ids.has(toolCall.callId));
}
```

- [ ] **Step 4: Use the matcher per message**

For each assistant message, compute its matched calls and pass them to `ChatMessageItem`. Do not select calls by message position or latest-assistant order.

- [ ] **Step 5: Run focused checks**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/tool-call-mapping.test.ts
pnpm build
```

Expected: TypeScript accepts the new prop path and the UI still renders ordinary text-only messages.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/chat/useChatController.ts frontend/src/app/page.tsx frontend/src/components/chat/ChatMessageList.tsx frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts
git commit -m "refactor: pass native tool calls to chat messages"
```

### Task 4: Render tool calls directly with AI Elements

**Files:**
- Modify: `frontend/src/components/chat/ChatMessageItem.tsx`
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/components/ai-elements/tool.tsx` only if its current props cannot represent documented lifecycle states
- Delete: `frontend/src/hooks/chat/tool-progress.ts` after callers are removed
- Modify: `frontend/src/lib/types/chat.ts` after `McpProgressEvent` callers are removed

**Interfaces:**
- Consumes: `AssembledToolCall[]` matched by message `toolCallIds`.
- Produces: AI Elements tool cards with running, finished, and error states.

- [ ] **Step 1: Add matched tool calls to ChatMessageItem props**

Add:

```ts
toolCalls: AssembledToolCall[];
```

Remove props and helpers that exist only for `McpProgressEvent`: `progressToolRuns`, `buildProgressToolRuns`, `ToolRunDisplay`, and `LegacyToolNotice`.

- [ ] **Step 2: Render one AI Elements card per call**

Map each matched call to the existing AI Elements primitives:

```tsx
<Tool key={toolCall.callId} defaultOpen>
  <ToolHeader
    type={`tool-\${toolCall.name}`}
    state={toolCall.status === "running"
      ? "input-available"
      : toolCall.status === "error"
        ? "output-error"
        : "output-available"}
  />
  <ToolContent>
    <ToolInput input={toolCall.input ?? toolCall.args} />
    <ToolOutput
      output={toolCall.output}
      errorText={toolCall.error}
    />
  </ToolContent>
</Tool>
```

Use the current AI Elements component prop types; if the installed source uses different state names, adapt the mapping to those exact types rather than adding a second UI abstraction.

- [ ] **Step 3: Keep app-specific UI separate**

Keep citations, feedback actions, retry/recovery buttons, and `MessageResponse` in the existing custom message shell. Do not move those concerns into the tool component.

- [ ] **Step 4: Remove the translation layer**

Delete `toolCallsToMcpEvents`, `withLiveToolProgress`, `McpProgressEvent`, `SdkToolProgress`, `stringifyToolPayload`, and `LegacyToolNotice` after `rg` confirms there are no callers. Do not replace them with another event format.

- [ ] **Step 5: Update tests**

Remove MCP-progress projection tests. Add component/helper tests for two calls on one message, calls split across two messages, running/finished/error states, and no tool cards for unrelated calls.

- [ ] **Step 6: Run focused checks**

Run:

```bash
cd frontend && pnpm test -- src/hooks/chat/__tests__/tool-call-mapping.test.ts src/hooks/chat/__tests__/message-projection.test.ts
pnpm lint
pnpm build
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/chat/ChatMessageItem.tsx frontend/src/components/chat/ChatMessageList.tsx frontend/src/components/ai-elements/tool.tsx frontend/src/hooks/chat/tool-progress.ts frontend/src/lib/types/chat.ts frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts frontend/src/hooks/chat/__tests__/message-projection.test.ts
git commit -m "refactor: render native tool calls with AI Elements"
```

### Task 5: Verify replay and live behavior

**Files:**
- Modify: `frontend/tests/e2e/chat-streaming.spec.ts`
- Modify: `frontend/tests/e2e/chat-live.spec.ts`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the direct `stream.toolCalls` rendering path.
- Produces: browser-level proof that tool calls remain associated with the correct assistant message.

- [ ] **Step 1: Add live assertions**

Verify that a running tool card appears before completion, updates to completed or error state, displays structured input/output, and remains attached to the correct assistant message.

- [ ] **Step 2: Add multi-call assertions**

Verify two simultaneous or sequential calls do not merge into one latest-assistant timeline and that each call ID renders once.

- [ ] **Step 3: Add replay assertions**

Reload or rejoin a durable thread and verify persisted assistant/tool messages do not duplicate. If the Agent Server does not replay live `toolCalls` for completed history, document that limitation and verify native persisted `tool_calls`/`ToolMessage` behavior instead; do not restore the custom fallback layer.

- [ ] **Step 4: Update the changelog**

Record that frontend tool rendering now follows the documented `AssembledToolCall` and AI Elements integration path.

- [ ] **Step 5: Run final checks**

Run:

```bash
cd frontend && pnpm lint
pnpm test
pnpm build
pnpm test:e2e -- tests/e2e/chat-streaming.spec.ts tests/e2e/chat-live.spec.ts
git diff --check
```

- [ ] **Step 6: Audit removed paths**

Run:

```bash
rg -n "McpProgressEvent|mcp_progress_events|toolCallsToMcpEvents|withLiveToolProgress|LegacyToolNotice|SdkToolProgress|stringifyToolPayload" frontend/src frontend/tests
git status --short
```

Expected: no active frontend references remain, except documentation/history explicitly describing the removed path.

- [ ] **Step 7: Commit**

```bash
git add frontend/tests/e2e/chat-streaming.spec.ts frontend/tests/e2e/chat-live.spec.ts CHANGELOG.md
git commit -m "test: verify native AI Elements tool rendering"
```

## Final verification checklist

- [ ] Each assistant message preserves its native tool-call IDs.
- [ ] Each `AssembledToolCall` renders only under its owning message.
- [ ] Running, finished, and error states come directly from `AssembledToolCall.status`.
- [ ] No `McpProgressEvent` translation remains in the frontend.
- [ ] AI Elements `MessageResponse`, `ToolInput`, and `ToolOutput` remain the rendering primitives.
- [ ] Citations, feedback, retry/recovery, durable threads, and streamed text remain unchanged.
- [ ] No `Conversation` or scroll behavior changes are mixed into this work.
- [ ] Frontend unit, build, lint, and relevant E2E checks pass.

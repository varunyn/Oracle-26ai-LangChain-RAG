# Frontend Ownership Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the chat frontend so `@langchain/react` is the only chat runtime owner, `@langchain/core` is limited to message models/types, and AI Elements remains a rendering layer instead of a second state system.

**Architecture:** Keep one runtime path from LangGraph Agent Server to `useStream`/selector hooks, then adapt only the minimum missing backend contract into UI-ready data. Remove duplicate projection/fallback logic in favor of one visible message path, one tool-activity path, and one AI Elements presentation layer.

**Tech Stack:** Next.js 16, React 19, TypeScript, `@langchain/react`, `@langchain/core`, AI Elements, Vitest, Playwright, LangGraph Agent Server

## Global Constraints

- `@langchain/react` owns chat transport, thread lifecycle, stream state, selector hooks, and tool-call assembly.
- `@langchain/core` is allowed only for message classes/types and test fixtures, not as a second frontend runtime.
- AI Elements components are presentational; they must not become a parallel source of chat state.
- Prefer native `stream.messages`, `stream.toolCalls`, `stream.values`, and `useChannel` before adding custom local state.
- If MCP activity cannot be represented as native tool calls, keep a single small adapter close to the stream provider instead of spreading custom logic across hooks and components.
- Do not add fallback or legacy code paths.
- Use `pnpm` for frontend commands.

---

## File Map

- `frontend/src/providers/langgraph-stream-provider.tsx`: source of truth for Agent Server stream hookup and any narrowly-scoped custom channel adapters.
- `frontend/src/hooks/chat/useChatController.ts`: derives UI-facing state from stream/runtime outputs; should not own transport details.
- `frontend/src/hooks/chat/useChatActions.ts`: submit/stop/retry/clear actions only.
- `frontend/src/hooks/chat/message-projection.ts`: canonical projection of visible messages; should stay small and deterministic.
- `frontend/src/components/chat/ChatMessageList.tsx`: AI Elements conversation rendering only.
- `frontend/src/components/chat/ChatMessageItem.tsx`: assistant/user message rendering only.
- `frontend/src/components/chat/McpActivityList.tsx`: custom MCP activity rendering if backend remains on a custom channel.
- `frontend/src/lib/types/mcp-activity.ts`: adapter from raw custom events to UI-ready MCP activity.
- `frontend/AGENTS.md`: frontend ownership rules and anti-patterns.
- `tests` under `frontend/src/hooks/chat` and `frontend/src/lib/types/__tests__`: focused unit coverage for projection and adapter behavior.

### Task 1: Lock Ownership Rules in Repo Docs

**Files:**
- Modify: `frontend/AGENTS.md`
- Modify: `CHANGELOG.md`
- Test: none

**Interfaces:**
- Consumes: current frontend architecture (`useStream`, `useChatController`, AI Elements components)
- Produces: explicit repo guidance for future frontend contributors

- [ ] **Step 1: Rewrite `frontend/AGENTS.md` ownership section**

Add a section that states:

```md
## Chat Ownership

- `@langchain/react` is the frontend chat runtime. It owns `useStream`, selector hooks, thread/run lifecycle, and assembled tool calls.
- `@langchain/core` is for message classes/types only. Do not build a second frontend runtime around it.
- AI Elements is the rendering layer. Keep message/tool/conversation components dumb and feed them state from `@langchain/react`.
```

- [ ] **Step 2: Add explicit anti-patterns**

Add bullets that forbid:

```md
- Building a second message store beside `stream.messages` / `stream.values`
- Treating AI Elements components as state managers
- Mixing native `stream.toolCalls` with separate ad hoc tool UI paths unless the backend contract truly differs
- Spreading custom channel parsing across provider, controller, and rendering components
```

- [ ] **Step 3: Record the guidance update in the changelog**

Add a short entry under `## 2026-06-29`:

```md
- Clarified frontend ownership so `@langchain/react` is the only chat runtime, `@langchain/core` is limited to message models/types, and AI Elements remains presentation-only.
```

- [ ] **Step 4: Review doc diff**

Run: `git diff -- frontend/AGENTS.md CHANGELOG.md`

Expected: only documentation changes describing runtime ownership and anti-patterns.

### Task 2: Inventory Runtime Overlap and Delete Targets

**Files:**
- Inspect: `frontend/src/providers/langgraph-stream-provider.tsx`
- Inspect: `frontend/src/hooks/chat/useChatController.ts`
- Inspect: `frontend/src/hooks/chat/useChatActions.ts`
- Inspect: `frontend/src/hooks/chat/message-projection.ts`
- Inspect: `frontend/src/components/chat/ChatMessageList.tsx`
- Test: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`

**Interfaces:**
- Consumes: current stream provider/controller/renderer responsibilities
- Produces: deletion list and ownership table for the cleanup refactor

- [ ] **Step 1: Write a short ownership table in the plan PR/notes**

Capture this target split:

```ts
type FrontendOwnership = {
  provider: "transport + stream selectors + minimal custom channel adapters";
  controller: "derived UI state only";
  actions: "submit/retry/stop/clear only";
  components: "render only";
};
```

- [ ] **Step 2: Identify code that duplicates runtime ownership**

Review for:

```ts
// delete or collapse if duplicated elsewhere
serverThreadMessages
authoritativeThreadMessages
custom message merging beyond one finalization rule
tool activity parsing outside provider/lib adapter
render-time state reconstruction
```

- [ ] **Step 3: Write failing checklist assertions before refactor**

Add or update tests to assert:

```ts
expect(sourceOfVisibleMessages).toBe("stream.messages + finalized state rule");
expect(sourceOfToolCalls).toBe("stream.toolCalls");
expect(sourceOfMcpActivity).toBe("provider channel adapter only");
```

- [ ] **Step 4: Run focused baseline tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS before cleanup starts, so later behavior changes are intentional.

### Task 3: Reduce the Provider to One Runtime Surface

**Files:**
- Modify: `frontend/src/providers/langgraph-stream-provider.tsx`
- Test: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`

**Interfaces:**
- Consumes: `useStream`, `useChannel`, LangGraph thread state fetches
- Produces:
  - `stream: ReturnType<typeof useStream>`
  - `mcpToolActivities: McpToolActivity[]`
  - minimal transport/thread-state error data

- [ ] **Step 1: Write a failing provider-focused test or note the contract**

Target contract:

```ts
type LangGraphStreamContextValue = {
  stream: StreamValue;
  mcpToolActivities: McpToolActivity[];
  transportError: Error | null;
  authoritativeThreadMessages?: BaseMessageWithKwargs[];
};
```

- [ ] **Step 2: Remove provider-owned state that duplicates stream state unless it is strictly for finalization**

Keep only state justified by one of:

```ts
"hydrate final authoritative thread state after successful completion"
"adapt custom:mcp_tool_activity into McpToolActivity[]"
"surface transport/thread hydration errors"
```

- [ ] **Step 3: Keep custom channel parsing local**

The provider should be the only place that directly does this:

```ts
const mcpActivityEvents = useChannel(stream, ["custom:mcp_tool_activity"]);
const mcpToolActivities = projectMcpToolActivities(mcpActivityEvents);
```

- [ ] **Step 4: Re-run focused tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS with no behavior regressions in visible-message or MCP-activity contracts.

### Task 4: Make the Controller Derived-State Only

**Files:**
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`

**Interfaces:**
- Consumes:
  - `stream.messages`
  - `stream.toolCalls`
  - `stream.values`
  - `mcpToolActivities`
- Produces:
  - `messages`
  - `toolCalls`
  - `mcpToolActivities`
  - `status`
  - `progress`

- [ ] **Step 1: Keep one visible-message derivation rule**

Target logic:

```ts
const liveMessages = projectStreamMessages({ streamMessages: stream.messages });
const finalizedMessages = stateMessages
  ? projectStreamMessages({ streamMessages: stateMessages })
  : undefined;
const messages = selectMessagesForStatus(liveMessages, finalizedMessages, status);
```

- [ ] **Step 2: Remove transport/provider logic from the controller**

Do not let the controller own:

```ts
"thread state fetching"
"custom channel parsing"
"message persistence reconciliation beyond one selection rule"
```

- [ ] **Step 3: Keep tool surfaces separate and explicit**

Target outputs:

```ts
toolCalls: stream.toolCalls ?? []
mcpToolActivities: provider.mcpToolActivities
```

- [ ] **Step 4: Run controller regression tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts`

Expected: PASS and no duplicate/hidden message regressions.

### Task 5: Keep AI Elements Purely Presentational

**Files:**
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/components/chat/ChatMessageItem.tsx`
- Modify: `frontend/src/components/chat/McpActivityList.tsx`
- Test: `frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts`

**Interfaces:**
- Consumes:
  - `messages`
  - `toolCalls`
  - `mcpToolActivities`
  - `status`
- Produces: rendered conversation only

- [ ] **Step 1: Keep Conversation/Message/Tool components free of runtime logic**

Allowed responsibilities:

```ts
"conditional rendering"
"mapping message props to AI Elements"
"showing tool states already assembled elsewhere"
```

- [ ] **Step 2: Forbid event parsing or stream inspection in components**

Disallowed example:

```ts
// do not do this in components
useChannel(...)
stream.submit(...)
fetch("/threads/...")
```

- [ ] **Step 3: Keep MCP activity as a render input, not a data source**

Target prop boundary:

```ts
<ChatMessageList
  messages={chat.messages}
  toolCalls={chat.toolCalls}
  mcpToolActivities={chat.mcpToolActivities}
/>
```

- [ ] **Step 4: Re-run focused UI-state mapping tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/tool-call-mapping.test.ts src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS with rendering still driven by pre-assembled state.

### Task 6: Decide the Long-Term MCP Contract

**Files:**
- Modify later: backend graph/runtime files if native tool-call normalization is chosen
- Modify later: `frontend/src/providers/langgraph-stream-provider.tsx` and `frontend/src/components/chat/McpActivityList.tsx` if custom adapter is removed
- Test: workflow tests and frontend tests

**Interfaces:**
- Consumes: current custom `custom:mcp_tool_activity` channel
- Produces: one supported long-term tool-activity contract

- [ ] **Step 1: Choose one of two contracts**

Option A:

```md
Backend normalizes MCP activity into native LangChain/LangGraph tool-call surfaces.
Frontend renders only `stream.toolCalls`.
```

Option B:

```md
Backend keeps `custom:mcp_tool_activity`.
Frontend keeps exactly one adapter in `langgraph-stream-provider.tsx` + `mcp-activity.ts`.
```

- [ ] **Step 2: Reject mixed ownership**

Do not keep this long-term:

```md
native `stream.toolCalls` for some tool paths + scattered custom MCP parsing for others in multiple files
```

- [ ] **Step 3: If Option A is chosen, delete `McpActivityList` path after parity is proven**

Validation command:

```bash
pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/tool-call-mapping.test.ts src/hooks/chat/__tests__/stream-message-contract.test.ts
```

- [ ] **Step 4: If Option B is chosen, document that exception in `frontend/AGENTS.md`**

Required note:

```md
`custom:mcp_tool_activity` is the only approved custom chat event adapter in the frontend, and it must remain provider-local.
```

### Task 7: Final Verification and Cleanup Gate

**Files:**
- Verify: frontend runtime + docs

**Interfaces:**
- Consumes: all prior cleanup tasks
- Produces: verified, documented frontend ownership model

- [ ] **Step 1: Run lint**

Run: `pnpm --dir frontend lint`

Expected: PASS

- [ ] **Step 2: Run focused unit coverage**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts src/hooks/chat/__tests__/tool-call-mapping.test.ts src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS

- [ ] **Step 3: Run full frontend build**

Run: `pnpm --dir frontend build`

Expected: PASS

- [ ] **Step 4: Manual runtime verification**

Run the local dev app on port `4040` and verify:

```md
- one answer per run
- citations visible without refresh
- native tool calls visible when backend emits native tool calls
- MCP activity visible only through the single approved adapter or native tool calls, depending on Task 6
- no duplicate message/tool rendering after replay or refresh
```

- [ ] **Step 5: Commit**

```bash
git add frontend/AGENTS.md CHANGELOG.md frontend/src/providers/langgraph-stream-provider.tsx frontend/src/hooks/chat/useChatController.ts frontend/src/components/chat/ChatMessageList.tsx frontend/src/components/chat/ChatMessageItem.tsx frontend/src/components/chat/McpActivityList.tsx frontend/src/lib/types/mcp-activity.ts frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts frontend/src/hooks/chat/__tests__/tool-call-mapping.test.ts frontend/src/lib/types/__tests__/mcp-activity.test.ts
git commit -m "refactor: clarify frontend chat ownership"
```

## Self-Review

- Spec coverage: this plan covers ownership rules, cleanup direction, AI Elements boundaries, MCP contract decision, testing, and documentation.
- Placeholder scan: all tasks contain exact files, commands, and expected outcomes.
- Type consistency: the plan consistently treats `stream` as the runtime source, `McpToolActivity[]` as the custom adapter output, and AI Elements props as presentational inputs only.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-frontend-ownership-cleanup.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

# AI Elements Conversation Scroll Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom chat scroll container and `useScrollToBottom` hook with AI Elements’ `Conversation`, `ConversationContent`, and `ConversationScrollButton` components without changing message, tool-call, citation, or stream behavior.

**Architecture:** AI Elements owns the scroll viewport, auto-scroll behavior, and scroll-to-latest affordance. `ChatMessageList` remains responsible for empty state, message mapping, tool-call matching, citations, feedback, and streaming indicators. The controller no longer creates or returns a DOM ref.

**Tech Stack:** Next.js 16, React 19, TypeScript, AI Elements source components, Vitest, Playwright, pnpm.

## Global Constraints

- Preserve `stream.messages`, native `AssembledToolCall` matching, citations, retries, recovery, and feedback.
- Replace only scroll ownership; do not redesign message cards or tool cards.
- Do not add a second scroll listener or retain `useScrollToBottom`.
- Keep the existing responsive width, padding, empty state, and message spacing.
- Use the official AI Elements source component as the baseline; adapt styling locally only where required by the existing layout.
- Update `CHANGELOG.md` under the current date.
- Do not commit or modify unrelated dirty-worktree files.

---

### Task 1: Establish the current scroll contract

**Files:**
- Modify: `frontend/src/hooks/__tests__/useScrollToBottom.test.ts` if it exists
- Create: `frontend/src/components/chat/__tests__/ChatMessageList.scroll.test.tsx` if component-level coverage is available
- Modify: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- Consumes: current `ChatMessageList) scroll behavior and `useScrollToBottom).
- Produces: regression coverage for auto-scroll, manual scroll-up behavior, and scroll-to-latest behavior.

- [ ] **Step 1: Inventory current test coverage**

Run:

```bash
rg -n "useScrollToBottom|scrollTo|scrollTop|scroll-height|scroll button|chat-message-list" frontend/src frontend/tests
```

Record the actual existing assertions before changing the hook.

- [ ] **Step 2: Add the minimum browser assertions**

Cover a streaming turn that appends content, a user scroll-up interaction, and the return-to-latest affordance. Keep selectors on stable test IDs rather than implementation-specific class names.

- [ ] **Step 3: Run the current focused checks**

Run:

```bash
cd frontend && pnpm exec vitest run src/hooks src/components/chat
pnpm exec playwright test tests/e2e/chat-streaming.spec.ts --grep "scroll|stream"
```

Expected: the current behavior is recorded before replacing the scroll implementation.

- [ ] **Step 4: Commit the baseline tests**

```bash
git add frontend/src/hooks/__tests__ frontend/src/components/chat/__tests__ frontend/tests/e2e/chat-streaming.spec.ts
git commit -m "test: capture chat scroll behavior before Conversation migration"
```

### Task 2: Add the official Conversation source component

**Files:**
- Create: `frontend/src/components/ai-elements/conversation.tsx`
- Modify: `frontend/src/components/chat/__tests__/ChatMessageList.scroll.test.tsx` if needed for the component contract

**Interfaces:**
- Consumes: child content and the existing chat viewport layout.
- Produces: `Conversation), `ConversationContent), and `ConversationScrollButton) exports with the official AI Elements behavior.

- [ ] **Step 1: Add the AI Elements component source**

Use the current AI Elements registry/source implementation corresponding to the official integration guide. The component must expose:

```tsx
<Conversation>
  <ConversationContent>{children}</ConversationContent>
  <ConversationScrollButton />
</Conversation>
```

Do not build a second custom scroll hook inside this file.

- [ ] **Step 2: Preserve required layout behavior**

Keep the component source editable and ensure the outer `Conversation` can receive the existing `className), ref behavior, and flex sizing required by the page layout. Do not add message-specific logic to the component.

- [ ] **Step 3: Run typecheck/build**

Run:

```bash
cd frontend && pnpm build
```

Expected: the new source component compiles with the project’s current React and Tailwind versions.

- [ ] **Step 4: Commit the source component**

```bash
git add frontend/src/components/ai-elements/conversation.tsx
git commit -m "feat: add AI Elements Conversation component"
```

### Task 3: Move ChatMessageList into Conversation

**Files:**
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/app/e2e/native-tool-calls/page.tsx`
- Modify: `frontend/src/components/chat/__tests__/ChatMessageList.scroll.test.tsx`

**Interfaces:**
- Consumes: existing `messages), `toolCalls), status, citations, and action callbacks.
- Produces: the same visible chat UI inside AI Elements’ managed scroll viewport.

- [ ] **Step 1: Replace the custom viewport**

Replace the current ref-bearing `div) and inner message wrapper with:

```tsx
<Conversation className="mx-auto flex w-full max-w-4xl flex-1 min-h-0">
  <ConversationContent className="overflow-x-hidden px-4 py-6 sm:px-6 sm:py-7">
    {emptyState}
    <div className="space-y-6">{messageRows}</div>
  </ConversationContent>
  <ConversationScrollButton />
</Conversation>
```

Keep the existing empty state, message rows, streaming indicator, tool-call matching, and error rendering unchanged.

- [ ] **Step 2: Remove the DOM ref prop**

Delete `chatContainerRef) from `ChatMessageListProps), the component destructuring, and the native E2E fixture usage. The Conversation component owns the scroll element.

- [ ] **Step 3: Preserve stable selectors**

Keep `data-testid="chat-message-list"` on the Conversation root and preserve `data-chat-status), `data-testid="chat-message-item"`, and tool-card selectors used by E2E tests.

- [ ] **Step 4: Run focused tests**

Run:

```bash
cd frontend && pnpm exec vitest run src/components/chat src/hooks/chat
pnpm build
```

Expected: typechecking passes and message/tool rendering remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/chat/ChatMessageList.tsx frontend/src/app/e2e/native-tool-calls/page.tsx frontend/src/components/chat/__tests__
git commit -m "refactor: use AI Elements Conversation for chat scrolling"
```

### Task 4: Remove the custom scroll hook and controller ref

**Files:**
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Modify: `frontend/src/hooks/chat/controller-types.ts) if the returned controller type includes the ref
- Delete: `frontend/src/hooks/useScrollToBottom.ts`
- Delete: `frontend/src/hooks/__tests__/useScrollToBottom.test.ts` if it only tests the deleted hook
- Modify: `frontend/src/app/page.tsx`

**Interfaces:**
- Consumes: controller state and message list props.
- Produces: no custom DOM scroll ref or scroll-effect dependency in the chat controller.

- [ ] **Step 1: Remove the hook import and call**

Delete the `useScrollToBottom) import and:

```ts
const chatContainerRef = useScrollToBottom(status, messages);
```

- [ ] **Step 2: Remove the returned ref**

Delete `chatContainerRef) from the controller return value and remove `chat.chatContainerRef) from `page.tsx).

- [ ] **Step 3: Remove obsolete tests and verify no callers**

Run:

```bash
rg -n "useScrollToBottom|chatContainerRef" frontend/src frontend/tests
```

Expected: no active frontend references remain.

- [ ] **Step 4: Run lint and unit tests**

Run:

```bash
cd frontend && pnpm lint
pnpm exec vitest run src/hooks/chat src/components/chat
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/chat/useChatController.ts frontend/src/hooks/chat/controller-types.ts frontend/src/hooks/useScrollToBottom.ts frontend/src/hooks/__tests__ frontend/src/app/page.tsx
git commit -m "refactor: remove custom chat scroll hook"
```

### Task 5: Verify browser behavior and document the cleanup

**Files:**
- Modify: `frontend/tests/e2e/chat-streaming.spec.ts`
- Modify: `frontend/tests/e2e/chat-live.spec.ts`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the Conversation-managed message viewport.
- Produces: browser-level proof that scrolling, streaming, tool cards, citations, and replay still work together.

- [ ] **Step 1: Verify streaming auto-scroll**

Assert that streamed assistant content remains visible as it grows when the user is at the latest position.

- [ ] **Step 2: Verify manual scroll-up behavior**

Scroll upward during a longer stream and assert that the viewport does not forcibly jump to the bottom. Assert that `ConversationScrollButton) appears and returns the user to the latest content.

- [ ] **Step 3: Verify non-scroll regressions**

Run the existing native tool-call, citation, retry, and replay assertions. Do not rewrite those tests to depend on internal Conversation implementation details.

- [ ] **Step 4: Update the changelog**

Record that the custom chat scroll hook was replaced with the AI Elements Conversation component while preserving chat rendering and stream behavior.

- [ ] **Step 5: Run final checks**

Run:

```bash
cd frontend && pnpm lint
pnpm exec vitest run
pnpm build
pnpm exec playwright test tests/e2e/chat-streaming.spec.ts tests/e2e/chat-live.spec.ts
git diff --check
```

Expected: lint, unit tests, build, and the selected browser suites pass.

- [ ] **Step 6: Final audit**

Run:

```bash
rg -n "useScrollToBottom|chatContainerRef|scrollTo\\(|addEventListener\\(\"scroll\"" frontend/src frontend/tests
git status --short
```

Expected: no custom chat scroll implementation remains; unrelated worktree changes are not staged.

- [ ] **Step 7: Commit**

```bash
git add frontend/tests/e2e/chat-streaming.spec.ts frontend/tests/e2e/chat-live.spec.ts CHANGELOG.md
git commit -m "test: verify Conversation-managed chat scrolling"
```

## Final verification checklist

- [ ] AI Elements `Conversation), `ConversationContent), and `ConversationScrollButton) own chat scrolling.
- [ ] `useScrollToBottom) is deleted and has no callers.
- [ ] `ChatMessageList) retains message, tool, citation, feedback, retry, and recovery behavior.
- [ ] Stable E2E selectors remain available.
- [ ] Streaming auto-scroll, scroll-up protection, and return-to-latest behavior pass.
- [ ] Lint, unit tests, build, and selected E2E tests pass.

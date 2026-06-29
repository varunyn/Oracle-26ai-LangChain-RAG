# Server-Owned Thread History Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make LangGraph server persistence the sole source of truth for conversation history while retaining only the active thread ID as a browser reopen pointer.

**Architecture:** Remove persisted sidebar summaries and local/server history merging from the frontend session store. Keep thread summaries in memory only, replace them from `threads.search(...)`, and keep `rag_agent_thread_id` only to reconnect the active conversation after reload. Successful deletion removes the server thread and invalidates the in-memory sidebar list; failed deletion leaves client state unchanged.

**Tech Stack:** Next.js 16, React 19, `useSyncExternalStore`, `@langchain/react`, LangGraph Agent Server, Vitest, Playwright mocks.

## Global Constraints

- LangGraph Agent Server/SQLite owns thread state and messages.
- The frontend must not persist conversation summaries or message data in localStorage.
- Preserve active thread reconnection, server-backed sidebar history, delete behavior, citations, streaming, and existing message ids.
- Use pnpm for frontend commands.
- Update current-date changelog documentation.

---

### Task 1: Lock the server-authoritative session contract

**Files:**
- Modify: `frontend/src/hooks/__tests__/useChatSession.test.ts`
- Modify: `frontend/src/hooks/useChatSession.ts`

**Interfaces:**
- `createInitialState()` returns an active `threadId`, an empty in-memory `threadHistory`, and hydration status.
- `mergeThreadHistory` is removed; `refreshThreadHistory` replaces history with `loadThreadHistory(client)`.
- `localStorage` contains only `THREAD_ID_STORAGE_KEY`.

- [ ] **Step 1: Write failing tests** for no local history rehydration and server replacement:
```ts
expect(createInitialState()).toMatchObject({
  threadId: "thread-from-server",
  threadHistory: [],
});
expect(
  replaceThreadHistory(
    [{ id: "server-thread", title: "Server", createdAt: 1, updatedAt: 2 }],
  ),
).toEqual([{ id: "server-thread", title: "Server", createdAt: 1, updatedAt: 2 }]);
```

- [ ] **Step 2: Run the focused test and verify it fails**:
```bash
pnpm --dir frontend exec vitest run src/hooks/__tests__/useChatSession.test.ts
```

- [ ] **Step 3: Remove the persisted history key and merge behavior.** Keep the active thread ID storage, but make session snapshots hold history only in memory. Replace refresh merging with the exact server result:
```ts
const loaded = await loadThreadHistory(client);
const snapshot = getSessionSnapshot();
writeSessionState({ ...snapshot, threadHistory: loaded, hydrated: true });
```

- [ ] **Step 4: Run focused session tests and verify they pass**:
```bash
pnpm --dir frontend exec vitest run src/hooks/__tests__/useChatSession.test.ts
```

---

### Task 2: Reconcile deletion and sidebar refresh

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/hooks/chat/useChatActions.ts`
- Modify: `frontend/src/hooks/chat/controller-types.ts`
- Modify: `frontend/src/hooks/useChatSession.ts`
- Modify: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- `clearSessionChat` accepts the deleted active thread ID explicitly.
- A successful clear removes that ID from the in-memory history and resets the active ID.
- The page refreshes server history after completed runs and after clear, but never merges prior local summaries back in.

- [ ] **Step 1: Add a failing regression assertion** that a deleted thread is absent after clear even if the mocked server search returns only current server threads.

- [ ] **Step 2: Run the targeted Playwright test and verify the new assertion fails**:
```bash
pnpm --dir frontend test:e2e --grep "clear"
```

- [ ] **Step 3: Pass the explicit thread ID through `ClearSessionChat`, remove the deleted item from in-memory history, and keep the existing transactional DELETE behavior.**

- [ ] **Step 4: Make the page refresh server history without local merging, including after an unbound new chat when appropriate.**

- [ ] **Step 5: Run the targeted e2e and session tests**:
```bash
pnpm --dir frontend test:e2e --grep "clear"
pnpm --dir frontend exec vitest run src/hooks/__tests__/useChatSession.test.ts
```

---

### Task 3: Remove stale contracts and update documentation

**Files:**
- Modify: `frontend/src/constants/chat.ts`
- Modify: `docs/CHAT_MEMORY_AND_SESSIONS.md`
- Modify: `docs/CHAT_STREAMING_PROTOCOL.md`
- Modify: `docs/TRACING.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Remove `CHAT_THREAD_HISTORY_STORAGE_KEY` and all references.**

- [ ] **Step 2: Document the final ownership contract: SQLite/Agent Server stores threads; localStorage stores only the active thread ID; sidebar history comes from `threads.search`.**

- [ ] **Step 3: Run literal searches to prove no stale local-history contract remains**:
```bash
rg -n "CHAT_THREAD_HISTORY_STORAGE_KEY|rag_agent_chat_threads|localStorage.*thread history|mergeThreadHistory" frontend docs
```

- [ ] **Step 4: Run documentation and frontend checks**:
```bash
pnpm --dir frontend exec vitest run src/hooks/__tests__/useChatSession.test.ts
pnpm --dir frontend lint
pnpm --dir frontend build
```

---

### Task 4: Review the final diff

- [ ] **Step 1:** Confirm only the server-authoritative session cleanup files are staged.
- [ ] **Step 2:** Confirm no generated directories, unrelated backend changes, or user documentation files are included.
- [ ] **Step 3:** Report tests and any remaining unrelated dirty files before committing.

# Chat Streaming Controller Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the chat streaming controller into focused modules without changing frontend behavior or backend contracts.

**Architecture:** Keep `frontend/src/hooks/chat/useChatController.ts` as the public orchestration hook, then extract stream configuration, reference parsing, tool progress conversion, message projection, debug helpers, and action callbacks into smaller modules. Preserve the existing submit payload, `useStream` wiring, chat controller return shape, and e2e-observed behavior.

**Tech Stack:** Next.js 16, React 19, TypeScript, `@langchain/react` `useStream`, Playwright

## Global Constraints

- Preserve the current FastAPI/LangGraph integration.
- Preserve recent LangGraph v1 stream behavior.
- Preserve the duplicate submitted prompt fix.
- Preserve RAG references, MCP progress display, suggestions, feedback, and recovery actions.
- Do not rewrite around the LangChain deployment-cookbook `js-next` sample.
- Do not move chat execution into Next route handlers.
- Do not change FastAPI routes, request payload contracts, or stream response contracts.
- Do not split `frontend/src/app/settings/page.tsx` in this implementation.
- Do not add a new frontend unit-test runner in this implementation.
- Do not redesign chat UI components.

---

### Task 1: Extract stream configuration and shared controller types

**Files:**
- Create: `frontend/src/hooks/chat/controller-types.ts`
- Create: `frontend/src/hooks/chat/stream-config.ts`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- Consumes: `FlowMode` from `frontend/src/hooks/useChatBodyParams.ts`
- Produces: `ToastApi`, `MessageLike`, `PendingUserMessage`, `ChatStatus`, `ReferencePayload`, `UseChatControllerArgs`, `resolveLanggraphApiUrl()`, `buildSubmitPayload()`

- [ ] **Step 1: Use existing e2e coverage as the regression test baseline**

Check the existing regression target in `frontend/tests/e2e/chat-streaming.spec.ts`:

```ts
test('does not render clicked suggestions as duplicate user messages', async ({ page }) => {
  const suggestion = 'Tell me about Oracle 26ai Database.'
  // existing mocked stream and assertion stay unchanged
})
```

- [ ] **Step 2: Run the targeted e2e test before refactoring**

Run: `pnpm --dir frontend test:e2e --grep "does not render clicked suggestions as duplicate user messages"`
Expected: PASS before any code movement

- [ ] **Step 3: Create shared controller types**

Add `frontend/src/hooks/chat/controller-types.ts` with the controller-local types now embedded in `useChatController.ts`:

```ts
import type { FlowMode } from "@/hooks/useChatBodyParams";
import type { ContextUsage, MessageReferences } from "@/lib/types/chat";

export type ToastApi = {
  error: (description: string, title?: string) => void;
  success: (description: string, title?: string) => void;
};

export type ReferencePayload = MessageReferences;

export type MessageLike = {
  id?: string;
  role?: string;
  content?: string;
  references?: ReferencePayload | null;
};

export type PendingUserMessage = MessageLike & {
  submittedMessageCount: number;
};

export type ChatStatus = "submitted" | "streaming" | "ready" | "error";

export type SendOverrides = {
  mode?: FlowMode;
};

export type ClearSessionChat = (helpers: {
  setMessages?: (value: MessageLike[] | ((prev: MessageLike[]) => MessageLike[])) => void;
  setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
  setContextUsage: (
    value: ContextUsage | null | ((prev: ContextUsage | null) => ContextUsage | null),
  ) => void;
}) => void;

export type UseChatControllerArgs = {
  selectedModel: string;
  threadId: string;
  sessionId: string;
  collectionName: string;
  enableReranker: boolean;
  enableTracing: boolean;
  flowMode: FlowMode;
  toast: ToastApi;
  clearSessionChat: ClearSessionChat;
};
```

- [ ] **Step 4: Create stream config helpers**

Add `frontend/src/hooks/chat/stream-config.ts`:

```ts
import { getClientApiBase } from "@/lib/api-base";
import type { FlowMode } from "@/hooks/useChatBodyParams";

type ChatBodyParams = {
  model: string;
  thread_id?: string;
  session_id?: string;
  collection_name?: string;
  enable_reranker: boolean;
  enable_tracing: boolean;
  mode: FlowMode;
};

export function resolveLanggraphApiUrl(): string {
  return `${getClientApiBase()}/api/langgraph`;
}

export function buildSubmitPayload(
  text: string,
  bodyParams: ChatBodyParams,
  mode: FlowMode,
) {
  return {
    messages: [{ type: "human", content: text }],
    model: bodyParams.model,
    session_id: bodyParams.session_id,
    collection_name: bodyParams.collection_name,
    enable_reranker: bodyParams.enable_reranker,
    enable_tracing: bodyParams.enable_tracing,
    mode,
    context: { ...bodyParams, mode },
    metadata: { ...bodyParams, mode },
    configurable: { ...bodyParams, mode },
  };
}
```

- [ ] **Step 5: Update `useChatController.ts` to import the new types and helpers**

Replace the duplicated local definitions and inline submit payload assembly with imports from:

```ts
import {
  type ChatStatus,
  type MessageLike,
  type PendingUserMessage,
  type ReferencePayload,
  type SendOverrides,
  type ToastApi,
  type UseChatControllerArgs,
} from "@/hooks/chat/controller-types";
import { buildSubmitPayload, resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";
```

- [ ] **Step 6: Re-run the targeted regression test**

Run: `pnpm --dir frontend test:e2e --grep "does not render clicked suggestions as duplicate user messages"`
Expected: PASS

### Task 2: Extract reference parsing and tool progress conversion

**Files:**
- Create: `frontend/src/hooks/chat/references.ts`
- Create: `frontend/src/hooks/chat/tool-progress.ts`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- Consumes: `MessageLike`, `ReferencePayload` from `controller-types.ts`; `ContextUsage`, `McpProgressEvent` from `frontend/src/lib/types/chat.ts`
- Produces: `normalizeContextUsage()`, `isSameContextUsage()`, `toReferences()`, `toReferencesFromRawMessage()`, `traceIdFromMessage()`, `referencePayloadFromMessage()`, `toolCallsToMcpEvents()`, `withLiveToolProgress()`

- [ ] **Step 1: Keep the existing values-driven rendering test as the red/green guard**

Use this existing e2e behavior from `frontend/tests/e2e/chat-streaming.spec.ts`:

```ts
test('does not render clicked suggestions as duplicate user messages', async ({ page }) => {
  // mocked values stream already exercises assistant message mapping
})
```

- [ ] **Step 2: Run the targeted e2e test before extraction**

Run: `pnpm --dir frontend test:e2e --grep "does not render clicked suggestions as duplicate user messages"`
Expected: PASS

- [ ] **Step 3: Move reference helpers into `references.ts`**

Extract these pure helpers from `useChatController.ts` into `frontend/src/hooks/chat/references.ts`:

```ts
export function normalizeContextUsage(raw: unknown): ContextUsage | undefined { /* existing logic */ }
export function isSameContextUsage(a: ContextUsage | null, b: ContextUsage): boolean { /* existing logic */ }
export function toReferences(message: BaseMessageWithKwargs): ReferencePayload | null { /* existing logic */ }
export function toReferencesFromRawMessage(rawMessage: unknown): ReferencePayload | null { /* existing logic */ }
export function traceIdFromMessage(message: MessageLike): string | undefined { /* existing logic */ }
export function referencePayloadFromMessage(message: MessageLike): ReferencePayload | null { /* existing logic */ }
```

- [ ] **Step 4: Move tool progress helpers into `tool-progress.ts`**

Extract these helpers into `frontend/src/hooks/chat/tool-progress.ts`:

```ts
export function stringifyToolPayload(value: unknown): string | null { /* existing logic */ }
export function toolCallsToMcpEvents(toolCalls: AssembledToolCall[]): McpProgressEvent[] { /* existing logic */ }
export function withLiveToolProgress(
  messages: MessageLike[],
  progressEvents: McpProgressEvent[],
): MessageLike[] { /* existing logic */ }
```

- [ ] **Step 5: Import the extracted helpers back into `useChatController.ts`**

Replace the local implementations with imports:

```ts
import {
  isSameContextUsage,
  normalizeContextUsage,
  referencePayloadFromMessage,
  toReferences,
  toReferencesFromRawMessage,
  traceIdFromMessage,
} from "@/hooks/chat/references";
import { toolCallsToMcpEvents, withLiveToolProgress } from "@/hooks/chat/tool-progress";
```

- [ ] **Step 6: Re-run the targeted e2e test**

Run: `pnpm --dir frontend test:e2e --grep "does not render clicked suggestions as duplicate user messages"`
Expected: PASS

### Task 3: Extract message projection and debug helpers

**Files:**
- Create: `frontend/src/hooks/chat/message-projection.ts`
- Create: `frontend/src/hooks/chat/stream-debug.ts`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- Consumes: `MessageLike`, `PendingUserMessage`, `ReferencePayload`, `ChatStatus`; helpers from `references.ts` and `tool-progress.ts`
- Produces: `projectStreamMessages()`, `normalizeStatus()`, `getLastUserMessageText()`, `createPendingUserMessage()`, `mergePendingUserMessage()`, `debugChatStream()`, debug flag helpers

- [ ] **Step 1: Use existing chat protocol rendering tests as the regression anchor**

Keep these existing behaviors in `frontend/tests/e2e/chat-streaming.spec.ts`:

```ts
test('does not render clicked suggestions as duplicate user messages', async ({ page }) => { /* existing */ })
test('shows locally known chat history and switches active threads', async ({ page }) => { /* existing */ })
```

- [ ] **Step 2: Run the targeted streaming tests before extraction**

Run: `pnpm --dir frontend test:e2e --grep "chat streaming"`
Expected: PASS or a known pre-existing environment failure unrelated to this refactor

- [ ] **Step 3: Extract message projection into `message-projection.ts`**

Move the message mapping and fallback logic into a single helper with the same current behavior:

```ts
export function projectStreamMessages(args: {
  streamMessages: BaseMessageWithKwargs[] | undefined;
  streamValues: unknown;
  pendingUserMessage: PendingUserMessage | null;
  liveToolProgressEvents: McpProgressEvent[];
}): MessageLike[] {
  // existing mapping, raw values fallback, assistant refs fallback,
  // pending user merge, and live tool progress merge
}
```

Also move:

```ts
export function normalizeStatus(rawStatus: unknown, isLoading: boolean, hasError: boolean): ChatStatus { /* existing logic */ }
export function getLastUserMessageText(messages: MessageLike[]): string { /* existing logic */ }
export function createPendingUserMessage(text: string, submittedMessageCount: number): PendingUserMessage { /* existing logic */ }
export function mergePendingUserMessage(
  messages: MessageLike[],
  pendingUserMessage: PendingUserMessage | null,
): MessageLike[] { /* existing logic */ }
```

- [ ] **Step 4: Extract debug helpers into `stream-debug.ts`**

Move the debug-only helpers:

```ts
export function isChatStreamDebugEnabled(): boolean { /* existing logic */ }
export function debugChatStream(event: ChatStreamDebugEvent, payload: Record<string, unknown>): void { /* existing logic */ }
```

plus the summary and flag helpers they depend on.

- [ ] **Step 5: Update `useChatController.ts` to consume `projectStreamMessages()`**

Replace the large inline `useMemo<MessageLike[]>` block with:

```ts
const messages = useMemo(
  () =>
    projectStreamMessages({
      streamMessages: streamMessages as BaseMessageWithKwargs[] | undefined,
      streamValues,
      pendingUserMessage,
      liveToolProgressEvents,
    }),
  [liveToolProgressEvents, pendingUserMessage, streamMessages, streamValues],
);
```

- [ ] **Step 6: Re-run the targeted streaming tests**

Run: `pnpm --dir frontend test:e2e --grep "chat streaming"`
Expected: PASS or unchanged pre-existing environment failure

### Task 4: Extract action callbacks and reduce `useChatController.ts` to orchestration

**Files:**
- Create: `frontend/src/hooks/chat/useChatActions.ts`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/tests/e2e/chat-streaming.spec.ts`

**Interfaces:**
- Consumes: projected `messages`, `threadId`, `stream`, `toast`, body params, `clearSessionChat`, and state setters from `useChatController.ts`
- Produces: `sendUserMessage()`, `handleSubmit()`, `handleRetry()`, `handleRecoverDirect()`, `handleRecoverRagOnly()`, `handleResumeTurn()`, `handleFeedback()`, `handleClearChat()`, `handleStopStream()`

- [ ] **Step 1: Use the current streaming e2e suite as the behavior contract**

Protect these existing flows:

```ts
test('does not render clicked suggestions as duplicate user messages', async ({ page }) => { /* existing */ })
test('shows locally known chat history and switches active threads', async ({ page }) => { /* existing */ })
```

- [ ] **Step 2: Run the targeted streaming tests before extraction**

Run: `pnpm --dir frontend test:e2e --grep "chat streaming"`
Expected: PASS or unchanged pre-existing environment failure

- [ ] **Step 3: Extract the action callbacks into `useChatActions.ts`**

Create a hook that receives controller dependencies and returns the existing action surface:

```ts
export function useChatActions(args: {
  bodyParams: ReturnType<typeof useChatBodyParams>;
  clearSessionChat: ClearSessionChat;
  input: string;
  messages: MessageLike[];
  setFeedbackSubmitted: React.Dispatch<React.SetStateAction<boolean>>;
  setFeedbackSubmittedMessageIndexes: React.Dispatch<React.SetStateAction<Set<number>>>;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  setMaxCitationsToShow: React.Dispatch<React.SetStateAction<number>>;
  setPendingUserMessage: React.Dispatch<React.SetStateAction<PendingUserMessage | null>>;
  setContextUsage: React.Dispatch<React.SetStateAction<ContextUsage | null>>;
  stream: ReturnType<typeof useStream>;
  streamMessageCount: number;
  threadId: string;
  toast: ToastApi;
}): { /* existing handlers */ } {
  // move existing callback bodies without changing behavior
}
```

- [ ] **Step 4: Reduce `useChatController.ts` to orchestration**

After importing `useChatActions`, `useChatController.ts` should primarily:

```ts
const bodyParams = useChatBodyParams(/* existing args */);
const stream = useStream(/* existing args */);
const liveToolProgressEvents = useMemo(/* existing tool call mapping */);
const messages = useMemo(/* projectStreamMessages */);
const status = normalizeStatus(/* existing args */);
const actions = useChatActions(/* controller dependencies */);
const { dynamicSuggestions, pendingSuggestion, suggestionsLoading, handleSuggestionClick } =
  useSuggestions(/* existing args */);
```

and return the same public controller shape currently consumed by `ChatInputBar` and `ChatMessageList`.

- [ ] **Step 5: Run build and e2e verification**

Run: `pnpm --dir frontend build`
Expected: exit 0

Run: `pnpm --dir frontend test:e2e`
Expected: all chat streaming tests pass, or any pre-existing environment failure is called out with exact command and symptom

- [ ] **Step 6: Review final file boundaries**

Confirm these files now each have a single clear responsibility:

- `useChatController.ts`
- `controller-types.ts`
- `stream-config.ts`
- `references.ts`
- `tool-progress.ts`
- `message-projection.ts`
- `stream-debug.ts`
- `useChatActions.ts`

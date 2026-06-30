# Suggestions Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link standalone suggestions traces to the parent chat session and record searchable request, model, and outcome metadata without changing suggestion behavior or masking local payloads.

**Architecture:** The frontend will include the current LangGraph `thread_id` in the suggestions request. The suggestions API will use that value as Langfuse `session_id`, derive request metadata from the existing request context, and update the root trace with a structured outcome and count. Existing Langfuse helper APIs remain the single tracing integration point.

**Tech Stack:** Next.js/React TypeScript, FastAPI/Pydantic, LangChain agents, Langfuse Python SDK, pytest, Playwright.

## Global Constraints

- Suggestions remain separate traces named `suggestions`.
- `thread_id` is used as Langfuse `session_id` only when present; `request_id` is never used as a session ID.
- Raw suggestion prompt data remains unmasked for this local non-production environment.
- Existing suggestion response and normalization behavior must remain unchanged.
- Workflow tests use fake model/agent boundaries and no real provider calls.
- Significant implementation changes must be recorded in `CHANGELOG.md`.

### Task 1: Propagate the chat thread ID from the frontend

**Files:**
- Modify: `frontend/src/hooks/useSuggestions.ts`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Test: `frontend/tests/e2e/chat-streaming.spec.ts` only if the request contract needs a browser assertion

**Interfaces:**
- `useSuggestions` consumes the existing `threadId` value owned by `useChatController`.
- `fetchSuggestions` adds optional `thread_id` to the existing JSON request body.
- The endpoint remains backward-compatible when `thread_id` is absent.

- [ ] **Step 1: Write the failing request-contract test or identify the existing browser route assertion.**

Add an assertion to the existing suggestions route interception that parses the request body and expects the active chat thread ID:

```ts
const body = JSON.parse(route.request().postData() ?? "{}");
expect(body.thread_id).toBe("thread-1");
```

Use the existing test fixture/thread setup; do not add a new chat state store.

- [ ] **Step 2: Run the focused frontend test to verify it fails.**

Run:

```bash
cd frontend && pnpm test:e2e tests/e2e/chat-streaming.spec.ts -g "suggestions"
```

Expected: the route assertion fails because the current request body has no `thread_id`.

- [ ] **Step 3: Implement the minimal propagation.**

Extend the `useSuggestions` props with `threadId: string | null`, pass it from `useChatController`, and extend `fetchSuggestions`:

```ts
function fetchSuggestions(
  lastMessage: string,
  lastUserMessage: string | null,
  selectedModel: string,
  threadId: string | null,
  onResult: (suggestions: string[]) => void,
  onDone: () => void
): void {
  // existing fetch setup
  body: JSON.stringify({
    last_message: lastMessage.slice(-4000),
    last_user_message: lastUserMessage?.slice(-2000) ?? undefined,
    model: selectedModel,
    thread_id: threadId || undefined,
  }),
}
```

Pass the same `threadId` to both automatic and manual suggestion fetches.

- [ ] **Step 4: Run the focused frontend test to verify it passes.**

Run:

```bash
cd frontend && pnpm test:e2e tests/e2e/chat-streaming.spec.ts -g "suggestions"
```

Expected: the suggestions tests pass and the intercepted body contains `thread_id`.

- [ ] **Step 5: Commit the frontend contract change.**

```bash
git add frontend/src/hooks/useSuggestions.ts frontend/src/hooks/chat/useChatController.ts frontend/tests/e2e/chat-streaming.spec.ts
git commit -m "feat(suggestions): propagate chat thread id"
```

### Task 2: Add Langfuse session, correlation, and outcome metadata

**Files:**
- Modify: `api/routes/suggestions.py`
- Modify: `api/middleware/request_context.py` only if an accessor is required; prefer the existing `REQUEST_ID_CTX`
- Test: `tests/workflow_tests/test_suggestions_api.py`

**Interfaces:**
- `SuggestionsRequest` gains `thread_id: str | None = None` with snake/camel aliases consistent with the existing request fields.
- `_generate_suggestions_async` consumes `thread_id` and the current request ID.
- `LangfuseChatTrace.update_output` receives `{"suggestion_count": int, "outcome": str}`.

- [ ] **Step 1: Write failing workflow assertions.**

Patch the Langfuse trace helpers in the workflow test and assert the generated request passes the thread ID into both trace creation and callback metadata:

```python
assert trace_kwargs["session_id"] == "thread-1"
assert callback_kwargs["session_id"] == "thread-1"
assert callback_kwargs["tags"] == [
    "feature:suggestions",
    "mode:suggestions",
    "model:xai.grok-4",
]
assert output == {"suggestion_count": 2, "outcome": "success"}
```

Add separate deterministic cases for an empty structured result and an exception, asserting `empty` and `error` respectively while preserving the HTTP response contract.

- [ ] **Step 2: Run the focused workflow tests to verify the new assertions fail.**

Run:

```bash
uv run pytest tests/workflow_tests/test_suggestions_api.py -q
```

Expected: the new capture assertions fail because the route currently passes `None` for session/thread data and only records `suggestion_count`.

- [ ] **Step 3: Implement request and trace metadata.**

Import `REQUEST_ID_CTX`, add the optional request field, and pass values through the existing trace helpers:

```python
request_id = REQUEST_ID_CTX.get()
trace_tags = [
    "feature:suggestions",
    "mode:suggestions",
    *( [f"model:{model_id}"] if model_id else [] ),
]
```

Use `thread_id` as `session_id` only when non-empty. Add `request_id` and `thread_id` to trace metadata through the existing callback `run_config["metadata"]` path. Keep prompt input unchanged.

- [ ] **Step 4: Implement explicit outcome updates.**

After normalization:

```python
outcome = "success" if suggestions else "empty"
langfuse_trace.update_output(
    {"suggestion_count": len(suggestions), "outcome": outcome}
)
```

On exceptions, update the trace before re-raising so the route still returns `[]`:

```python
langfuse_trace.update_output({"suggestion_count": 0, "outcome": "error"})
```

If the provider response exposes `finish_reason == "length"`, use `truncated`; otherwise do not infer truncation from token counts.

- [ ] **Step 5: Run the focused workflow tests to verify they pass.**

Run:

```bash
uv run pytest tests/workflow_tests/test_suggestions_api.py -q
```

Expected: all suggestions workflow tests pass.

- [ ] **Step 6: Commit the backend tracing change.**

```bash
git add api/routes/suggestions.py tests/workflow_tests/test_suggestions_api.py
git commit -m "feat(observability): enrich suggestions traces"
```

### Task 3: Verify the complete trace contract

**Files:**
- Modify: `CHANGELOG.md`
- Test: `tests/unit_tests/test_langfuse_tracing.py` if helper metadata coverage is needed

**Interfaces:**
- The live `/api/suggestions` request returns the existing `suggestions` response.
- Langfuse receives a standalone `suggestions` trace linked by `sessionId` when `thread_id` is supplied.

- [ ] **Step 1: Add the changelog entry.**

Add under the current date:

```markdown
- Linked suggestions traces to chat sessions and added request/outcome metadata for Langfuse debugging.
```

- [ ] **Step 2: Run focused static and test checks.**

Run:

```bash
uv run ruff check api/routes/suggestions.py tests/workflow_tests/test_suggestions_api.py
uv run pytest tests/workflow_tests/test_suggestions_api.py tests/unit_tests/test_langfuse_tracing.py -q
cd frontend && pnpm lint
```

Expected: exit code 0 for each command.

- [ ] **Step 3: Trigger a live suggestion request.**

Use the existing frontend or API surface with a realistic business prompt and a known thread ID. Do not use a placeholder arithmetic request.

- [ ] **Step 4: Inspect the resulting trace with the Langfuse CLI.**

Run:

```bash
langfuse --env .env api traces list --limit 5 --json
langfuse --env .env api traces get TRACE_ID_FROM_THE_PREVIOUS_COMMAND --fields core,io,observations,metrics --json
```

Verify: trace name `suggestions`, session ID equal to the chat thread ID, `feature:suggestions` and `mode:suggestions` tags, request/thread/model metadata, and output containing `suggestion_count` plus `outcome`.

- [ ] **Step 5: Commit verification documentation.**

```bash
git add CHANGELOG.md
git commit -m "docs: record suggestions observability changes"
```

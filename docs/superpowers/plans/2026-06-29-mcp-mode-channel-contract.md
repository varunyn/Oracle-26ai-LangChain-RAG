# MCP Mode Channel Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing LangGraph chat stream contract explicit and reliable for direct, RAG, MCP, and mixed runs without adding another frontend transport or state pipeline.

**Architecture:** Use the native LangGraph projections according to their documented meaning: `stream.messages` for live model-message streams, `stream.values` for graph-state snapshots, and `stream.toolCalls` for native tool execution lifecycle. Keep MCP progress in the existing named custom channel `custom:mcp_tool_activity`, consumed once by the stream provider and projected into presentational MCP activity cards. Preserve the app's thread-state hydration as the authority for replay/final message state.

**Tech Stack:** Next.js 16, React 19, TypeScript, `@langchain/react`, `@langchain/core`, AI Elements, Python, LangGraph Agent Server, Vitest, pytest

## Global Constraints

- Do not add mode-specific frontend state pipelines.
- `rag` must not gain a custom MCP/tool channel just because retrieval work happens.
- `custom:mcp_tool_activity` is the only approved custom chat event adapter in the frontend.
- `stream.toolCalls` must remain reserved for actual native LangGraph/LangChain tool-call events.
- MCP activity must include enough metadata to support multiple configured MCP servers.
- `stream.messages` is live model-message output, not the authoritative replay transcript.
- `stream.values` contains graph-state snapshots, not necessarily the final hydrated thread state.
- Server identity must come from configured MCP-server metadata or the adapter's configured prefix mapping, never from splitting an arbitrary tool name.
- Do not add fallback or legacy code paths.
- Record contract and architecture guidance in `frontend/AGENTS.md` and `CHANGELOG.md`.

## Documentation basis

Before implementation, verify the current official documentation for:

- [LangChain event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming#event-streaming): meanings of `stream.messages`, `stream.values`, and `stream.toolCalls`.
- [LangGraph custom stream channels](https://docs.langchain.com/oss/python/langgraph/frontend/custom-stream-channels): named `custom:<name>` channels, `useChannel`, raw event envelopes, and buffering.

Use the repository's installed package versions; do not infer APIs from older examples.

## Current implementation to preserve

The worktree already contains the initial custom-channel path in `mcp_activity.py`, the MCP and mixed graph nodes, `langgraph-stream-provider.tsx`, `mcp-activity.ts`, and `McpActivityList.tsx`. Treat those as existing implementation and make only the contract corrections below. Do not overwrite unrelated dirty-worktree changes.

---

## File Map

- `src/rag_agent/runtime/mcp_activity.py`: canonical custom MCP activity payload shape.
- `src/rag_agent/infrastructure/mcp_adapter_runtime.py`: configured server keys and the `tool_name_prefix=True` namespace relationship.
- `src/rag_agent/infrastructure/mcp_agent_executor.py`: source of MCP execution lifecycle events.
- `src/rag_agent/graphs/nodes/mcp.py`: MCP mode graph node stream emission.
- `src/rag_agent/graphs/nodes/mixed.py`: mixed mode graph node stream emission.
- `frontend/src/lib/types/mcp-activity.ts`: frontend adapter from raw custom events to UI state.
- `frontend/src/providers/langgraph-stream-provider.tsx`: single approved custom MCP channel consumer.
- `frontend/src/components/chat/McpActivityList.tsx`: MCP activity rendering.
- `frontend/AGENTS.md`: frontend ownership + mode/channel contract rules.
- `CHANGELOG.md`: repo-level record of the contract.
- `tests/unit_tests/test_mcp_activity.py`: backend contract tests.
- `frontend/src/lib/types/__tests__/mcp-activity.test.ts`: frontend contract tests.
- `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`: frontend visible-message/tool contract tests.

### Task 1: Document the Stable Mode/Channel Contract

**Files:**

- Modify: `frontend/AGENTS.md`
- Modify: `CHANGELOG.md`
- Test: none

**Interfaces:**

- Consumes: current frontend ownership rules
- Produces: explicit contract for future features and bug fixes

- [ ] **Step 1: Add a mode/channel contract section to `frontend/AGENTS.md`**

Add this section:

```md
## Mode/Channel Contract

- `direct`: native message stream only; no MCP activity channel
- `rag`: native message/state stream plus citations; retrieval is not MCP activity
- `mcp`: MCP activity uses `custom:mcp_tool_activity` when an MCP tool actually runs
- `mixed`: retrieval/citations use the normal state/message contract; MCP activity uses `custom:mcp_tool_activity` only when an MCP tool actually runs

- `stream.messages`: live model-message streams and message content deltas
- `stream.values`: graph-state snapshots during a run
- hydrated LangGraph thread state: authoritative replay/final message state
- `stream.toolCalls`: native LangGraph/LangChain tool execution lifecycle only
- `custom:mcp_tool_activity`: MCP-specific lifecycle progress, consumed once by the stream provider

Do not infer server identity by splitting a tool name. The backend must send the configured server key when it can resolve one.
```

- [ ] **Step 2: Add anti-pattern bullets**

Add these bullets:

```md
- Do not create a separate channel contract per mode
- Do not route RAG retrieval through custom tool-activity channels
- Do not convert MCP activity into fake native `stream.toolCalls`
- Do not treat `stream.messages` or an intermediate `stream.values` snapshot as replay authority
```

- [ ] **Step 3: Record the contract in `CHANGELOG.md`**

Add this entry under `## 2026-06-29`:

```md
- Clarified the chat stream contract: native message/state/tool projections retain their documented meanings, MCP progress uses one custom channel, and configured MCP server identity is propagated explicitly instead of inferred from arbitrary tool names.
```

- [ ] **Step 4: Review documentation diff**

Run: `git diff -- frontend/AGENTS.md CHANGELOG.md`

Expected: only the new contract language and changelog entry.

### Task 2: Expand the MCP Activity Payload for Multi-Server MCP

**Files:**

- Modify: `src/rag_agent/infrastructure/mcp_adapter_runtime.py`
- Modify: `src/rag_agent/runtime/mcp_activity.py`
- Modify: `src/rag_agent/infrastructure/mcp_agent_executor.py`
- Test: `tests/unit_tests/test_mcp_activity.py`
- Test: `tests/unit_tests/test_mcp_agent_executor.py`

**Interfaces:**

- Consumes: raw MCP executor lifecycle events
- Produces:
  - custom event name: `"mcp_tool_activity"`
  - payload fields:
    - `tool_run_id: str`
    - `tool_name: str`
    - `server_name: str | None`
    - `status: "running" | "finished" | "error"`
    - `args: object | None`
    - `output: object | None`
    - `error: str | None`

- [ ] **Step 1: Write the failing backend test**

Add this test:

```python
def test_mcp_tool_activity_event_preserves_server_name() -> None:
    event = mcp_tool_activity_event(
        {
            "phase": "start",
            "tool_run_id": "call-1",
            "tool_name": "Calculator_linear_regression",
            "server_name": "calculator",
            "args": {"data": [[1, 2], [2, 3.5]]},
        }
    )
    assert event["payload"]["server_name"] == "calculator"
```

- [ ] **Step 2: Run the failing backend test**

Run: `uv run pytest tests/unit_tests/test_mcp_activity.py -q`

Expected: FAIL because `server_name` is missing from the emitted payload.

- [ ] **Step 3: Implement the minimal payload change**

Update `mcp_tool_activity_event(...)` to emit:

```python
return {
    "name": MCP_TOOL_ACTIVITY_NAME,
    "payload": {
        "tool_run_id": str(event.get("tool_run_id") or ""),
        "tool_name": str(event.get("tool_name") or "unknown_tool"),
        "server_name": str(event.get("server_name") or "").strip() or None,
        "status": status,
        "args": event.get("args"),
        "output": event.get("result"),
        "error": event.get("error"),
    },
}
```

- [ ] **Step 4: Thread configured server metadata into emitted tool progress events**

In `mcp_agent_executor.py`, extend the callback event shape to include:

```python
{
    "phase": "start",
    "tool_name": tool_name,
    "server_name": _server_name_for_tool(tool_name, configured_server_keys),
    "args": getattr(call, "input", None),
    "tool_run_id": tool_call_id,
}
```

Use only configured keys from the adapter's `MultiServerMCPClient(..., tool_name_prefix=True)` namespace. Do not split an arbitrary tool name or treat an unknown prefix as a server. Add tests for duplicate tool names across configured servers and for an unrecognized name returning `None`.

- [ ] **Step 5: Run backend verification**

Run: `uv run pytest tests/unit_tests/test_mcp_activity.py tests/unit_tests/test_mcp_agent_executor.py -q`

Expected: PASS

### Task 3: Lock the Frontend MCP Activity Projection Contract

**Files:**

- Modify: `frontend/src/lib/types/mcp-activity.ts`
- Verify: `frontend/src/providers/langgraph-stream-provider.tsx`
- Verify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/lib/types/__tests__/mcp-activity.test.ts`
- Test: `frontend/src/lib/types/__tests__/mcp-activity.test.ts`

**Interfaces:**

- Consumes: raw `custom:mcp_tool_activity` events
- Produces:
  - `type McpToolActivity = { toolRunId, toolName, serverName, status, args, output, error }`

- [ ] **Step 1: Write the failing frontend test**

Add this test:

```ts
it("keeps server metadata and merges lifecycle updates by tool run id", () => {
  expect(
    projectMcpToolActivities([
      {
        method: "custom",
        params: {
          data: {
            name: "mcp_tool_activity",
            payload: {
              tool_run_id: "call-1",
              tool_name: "calculator_linear_regression",
              server_name: "calculator",
              status: "running",
              args: {
                data: [
                  [1, 2],
                  [2, 3.5],
                ],
              },
            },
          },
        },
      },
      {
        method: "custom",
        params: {
          data: {
            name: "mcp_tool_activity",
            payload: {
              tool_run_id: "call-1",
              tool_name: "calculator_linear_regression",
              server_name: "calculator",
              status: "finished",
              output: "ok",
            },
          },
        },
      },
    ]),
  ).toEqual([
    {
      toolRunId: "call-1",
      toolName: "Calculator_linear_regression",
      serverName: "calculator",
      status: "finished",
      args: {
        data: [
          [1, 2],
          [2, 3.5],
        ],
      },
      output: "ok",
      error: null,
    },
  ]);
});
```

- [ ] **Step 2: Run the failing frontend test**

Run: `pnpm --dir frontend exec vitest run src/lib/types/__tests__/mcp-activity.test.ts`

Expected: FAIL because `serverName` is not yet part of the frontend type/adapter.

- [ ] **Step 3: Implement the minimal adapter change**

Update `McpToolActivity` and `activityFromPayload(...)`. Normalize only non-empty strings to a server name. Preserve the first non-null `args`, and apply the latest lifecycle `status`, `output`, and `error`.

```ts
export type McpToolActivity = {
  toolRunId: string;
  toolName: string;
  serverName: string | null;
  status: McpToolActivityStatus;
  args: unknown;
  output: unknown;
  error: string | null;
};
```

and

```ts
serverName:
  typeof value.server_name === "string" && value.server_name.trim().length > 0
    ? value.server_name
    : null,
```

- [ ] **Step 4: Run frontend adapter verification**

Run: `pnpm --dir frontend exec vitest run src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS

### Task 4: Render Multi-MCP Activity Without Changing the Main Message Pipeline

**Files:**

- Modify: `frontend/src/components/chat/McpActivityList.tsx`
- Verify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`
- Test: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`

**Interfaces:**

- Consumes: `McpToolActivity[]`
- Produces: MCP activity cards labeled with server + tool while leaving message/tool-call pipelines unchanged

- [ ] **Step 1: Write the failing contract test**

Add this test:

```ts
it("keeps MCP activity separate from native tool-call projection", () => {
  const activities = projectMcpToolActivities([
    {
      method: "custom",
      params: {
        data: {
          name: "mcp_tool_activity",
          payload: {
            tool_run_id: "call-1",
            tool_name: "calculator_linear_regression",
            server_name: "calculator",
            status: "finished",
            output: "ok",
          },
        },
      },
    },
  ]);
  expect(activities[0]).toMatchObject({
    serverName: "calculator",
    toolName: "Calculator_linear_regression",
  });
});
```

- [ ] **Step 2: Run the failing contract test**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts`

Expected: FAIL until the test fixture and UI-facing type are updated consistently.

- [ ] **Step 3: Render server + tool identity in `McpActivityList.tsx`**

Change the rendered type/header identity to:

```tsx
const toolType =
  activity.serverName != null
    ? `mcp-${activity.serverName}-${activity.toolName}`
    : `mcp-${activity.toolName}`;
```

and include a visible label like:

```tsx
<div className="text-xs text-muted-foreground">
  {activity.serverName
    ? `${activity.serverName} / ${activity.toolName}`
    : activity.toolName}
</div>
```

without moving parsing or stream logic into the component.

- [ ] **Step 4: Run focused frontend contract tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts src/lib/types/__tests__/mcp-activity.test.ts`

Expected: PASS

### Task 5: Prove the Mode/Channel Rules in Tests

**Files:**

- Modify: `frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts`
- Modify: `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`
- Modify only if a workflow-level mode assertion is required: `tests/workflow_tests/test_chat_nonstream_and_validation.py`

**Interfaces:**

- Consumes:
  - `direct` mode
  - `rag` mode
  - `mcp` mode
  - `mixed` mode
- Produces explicit guardrails:
  - `rag` does not depend on MCP custom activity
  - `mcp` and `mixed` emit MCP activity only when MCP is used

- [ ] **Step 1: Add a frontend test note for RAG**

Add an assertion-oriented test or contract fixture that states:

```ts
expect(
  selectMessagesForStatus(liveMessages, finalizedMessages, "ready"),
).toEqual(finalizedMessages);
// no MCP activity required for citations/finalized RAG answers
```

- [ ] **Step 2: Add or update backend node tests**

Add deterministic tests that capture the node stream writer and prove:

```python
def test_mcp_node_emits_mcp_activity_events() -> None:
    emitted = run_mcp_node_with_fake_tool_and_captured_writer()
    assert [event["name"] for event in emitted] == ["mcp_tool_activity", "mcp_tool_activity"]
    assert [event["payload"]["status"] for event in emitted] == ["running", "finished"]


def test_mixed_node_emits_mcp_activity_only_for_mcp_turns() -> None:
    retrieval_only = run_mixed_node_with_fake_retrieval_and_captured_writer()
    mcp_turn = run_mixed_node_with_fake_mcp_tool_and_captured_writer()
    assert retrieval_only == []
    assert mcp_turn[0]["name"] == "mcp_tool_activity"


def test_rag_node_does_not_emit_mcp_activity_for_retrieval_only() -> None:
    emitted = run_rag_node_with_fake_retrieval_and_captured_writer()
    assert emitted == []
```

Implement the test-local setup directly in each test using the existing fake model/tool fixtures in `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`. Capture the stream-writer/callback boundary and return the collected event dictionaries; do not contact MCP, Oracle, or an LLM provider.

- [ ] **Step 3: Run focused backend tests**

Run: `uv run pytest tests/unit_tests/test_langgraph_mcp_mixed_nodes.py -q`

Expected: PASS

- [ ] **Step 4: Run focused frontend tests**

Run: `pnpm --dir frontend exec vitest run src/hooks/chat/__tests__/stream-message-contract.test.ts`

Expected: PASS

### Task 6: Final Verification and Commit

**Files:**

- Verify all touched contract, adapter, provider, rendering, test, and docs files
- Do not stage unrelated existing worktree changes, generated directories, `.playwright-mcp/`, `.pnpm-store/`, or `node_modules/`

**Interfaces:**

- Consumes: all previous tasks
- Produces: stable stream contract for future features

- [ ] **Step 1: Run lint**

Run: `pnpm --dir frontend lint`

Expected: PASS

- [ ] **Step 2: Run frontend build**

Run: `pnpm --dir frontend build`

Expected: PASS

- [ ] **Step 3: Run focused frontend tests**

Run: `pnpm --dir frontend exec vitest run src/lib/types/__tests__/mcp-activity.test.ts src/hooks/chat/__tests__/stream-message-contract.test.ts src/hooks/chat/__tests__/tool-call-mapping.test.ts`

Expected: PASS

- [ ] **Step 4: Run focused backend tests**

Run: `uv run pytest tests/unit_tests/test_mcp_activity.py tests/unit_tests/test_mcp_agent_executor.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py -q`

Expected: PASS

- [ ] **Step 5: Manual runtime verification**

Verify on local frontend port `4040`:

```md
- `rag` answers show citations without MCP activity cards
- `mcp` answers show MCP activity cards with server + tool identity
- `mixed` answers show citations plus MCP activity only when MCP is actually used
- native LangGraph/LangChain tool calls still render through `stream.toolCalls`
- replay hydration does not duplicate the visible final answer
```

When the Agent Server and configured test MCP server are unavailable, report the deterministic checks that ran and leave live stream verification explicitly unproven.

- [ ] **Step 6: Commit**

```bash
git add frontend/AGENTS.md CHANGELOG.md src/rag_agent/infrastructure/mcp_adapter_runtime.py src/rag_agent/runtime/mcp_activity.py src/rag_agent/infrastructure/mcp_agent_executor.py frontend/src/providers/langgraph-stream-provider.tsx frontend/src/lib/types/mcp-activity.ts frontend/src/components/chat/McpActivityList.tsx frontend/src/lib/types/__tests__/mcp-activity.test.ts frontend/src/hooks/chat/__tests__/stream-message-contract.test.ts tests/unit_tests/test_mcp_activity.py tests/unit_tests/test_mcp_agent_executor.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py
git commit -m "refactor: standardize mcp stream channel contract"
```

## Self-Review

- Spec coverage: this plan covers the contract definition, multi-MCP payload shape, frontend adapter, UI rendering, mode/channel guardrails, and verification.
- Placeholder scan: every task includes exact files, commands, and expected outputs.
- Type consistency: the plan consistently uses `server_name` in backend payloads, `serverName` in frontend types, `stream.toolCalls` only for native tool calls, and `custom:mcp_tool_activity` only for MCP activity.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-mcp-mode-channel-contract.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

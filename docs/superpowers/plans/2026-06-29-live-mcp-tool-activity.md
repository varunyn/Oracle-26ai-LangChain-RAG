# Live MCP Tool Activity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution selected).

**Goal:** Display live MCP tool lifecycle activity in the chat UI using a canonical LangGraph custom stream channel while preserving final replay metadata.

**Architecture:** Internal MCP execution remains inside the graph node, but its tool-progress callback is adapted to LangGraph's `get_stream_writer()`. The backend emits a named custom extension `mcp_tool_activity` with normalized started/completed/error events. The frontend subscribes to `custom:mcp_tool_activity` with `useChannel` and renders those events through a dedicated MCP activity projection; native `stream.toolCalls` remains reserved for actual Agent Server-native tool calls.

**Tech Stack:** LangGraph `get_stream_writer`, Agent Server custom SSE events, `@langchain/react` `useChannel`, React/TypeScript, Vitest, pytest.

## Global Constraints

- Do not synthesize native `stream.toolCalls` or `AIMessage.tool_calls` from final metadata.
- Preserve `mcp_tools_used` and `mcp_tool_invocations` in final assistant metadata for replay/history.
- Use one stable custom event name and one typed event schema across backend and frontend.
- Do not add a second chat transport or frontend proxy.
- Verify backend event emission, frontend projection, and the rendered activity path separately.

---

### Task 1: Define and test the canonical MCP activity event

**Files:**
- Create: `src/rag_agent/runtime/mcp_activity.py`
- Create: `frontend/src/lib/types/mcp-activity.ts`
- Test: `tests/unit_tests/test_mcp_activity.py`
- Test: `frontend/src/lib/types/__tests__/mcp-activity.test.ts`

**Event contract:**
```json
{
  "name": "mcp_tool_activity",
  "payload": {
    "tool_run_id": "call-1",
    "tool_name": "lookup",
    "status": "running|finished|error",
    "args": {},
    "output": {},
    "error": null
  }
}
```

- [ ] Write failing normalization tests for start, success, and error invocation payloads.
- [ ] Run the focused pytest/Vitest tests and verify they fail.
- [ ] Implement shared backend normalization helpers and matching frontend type guards.
- [ ] Run the focused tests and verify they pass.

### Task 2: Emit live custom events from MCP and mixed graph nodes

**Files:**
- Modify: `src/rag_agent/graphs/nodes/mcp.py`
- Modify: `src/rag_agent/graphs/nodes/mixed.py`
- Modify: `src/rag_agent/runtime/mcp_turn.py`
- Test: `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`

- [ ] Add a graph-node writer adapter using `get_stream_writer()`.
- [ ] Pass the adapter as `tool_progress_callback` to `run_mcp_agent_turn` for MCP and mixed modes.
- [ ] Keep the existing invocation collection and final assistant metadata unchanged.
- [ ] Assert that started/finished/error custom payloads are emitted while the fake MCP turn runs.
- [ ] Run the focused backend tests and verify the event contract.

### Task 3: Subscribe to the custom channel in the frontend

**Files:**
- Modify: `frontend/src/providers/langgraph-stream-provider.tsx`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Modify: `frontend/src/hooks/chat/controller-types.ts`
- Test: `frontend/src/hooks/chat/__tests__/mcp-activity-projection.test.ts`

- [ ] Add `useChannel(stream, ["custom:mcp_tool_activity"])` at the controller/provider boundary.
- [ ] Convert raw custom events into typed `McpToolActivity[]`, ignoring unrelated events and malformed payloads.
- [ ] Scope displayed activities to the active stream lifecycle and preserve the final metadata path for replay.
- [ ] Add deterministic projection tests for event ordering and status updates.
- [ ] Run focused frontend tests and lint.

### Task 4: Render dedicated MCP activity cards

**Files:**
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/components/chat/ChatMessageItem.tsx` or create `frontend/src/components/chat/McpActivityList.tsx`
- Test: `frontend/src/components/chat/__tests__/McpActivityList.test.tsx` if the existing test setup supports component tests.

- [ ] Render MCP activity from the dedicated activity prop, not from `stream.toolCalls`.
- [ ] Show tool name, running/completed/error status, arguments, and output/error.
- [ ] Keep native tool cards unchanged for native Agent Server tool calls.
- [ ] Ensure MCP activity does not create duplicate assistant messages or duplicate final answers.

### Task 5: Verify the full contract and document it

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/CHAT_STREAMING_PROTOCOL.md`
- Modify: `docs/CHAT_MEMORY_AND_SESSIONS.md`

- [ ] Run:
```bash
./.venv/bin/pytest tests/unit_tests/test_mcp_activity.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py tests/workflow_tests/test_langgraph_chat_contract.py -q
pnpm --dir frontend exec vitest run src/lib/types/__tests__/mcp-activity.test.ts src/hooks/chat/__tests__/mcp-activity-projection.test.ts
pnpm --dir frontend lint
pnpm --dir frontend build
```
- [ ] Verify a live MCP run emits custom activity events and one final assistant answer.
- [ ] Document that `stream.toolCalls` is native Agent Server activity and `mcp_tool_activity` is the explicit internal-MCP activity channel.


# MCP Mixed-Mode Tool Streaming Sub-graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-node MCP tool loop in mixed mode with a proper LangGraph sub-graph for real-time tool/LLM streaming.

**Architecture:** Build a `StateGraph` sub-graph with `call_llm`, `run_tools`, and `route` nodes. Tools are loaded in a setup node (`run_mixed_mcp_setup`) and stored in `runtime.context` for sub-graph node access. The sub-graph is registered at compile time in `chat_agent.py`. The `mixed_compose` node adapts to read sub-graph output from parent state messages.

**Tech Stack:** LangGraph 1.2.6, LangChain, Python 3.11, TypeScript (frontend type)

## Global Constraints

- LangGraph 1.2.6 minimum (pinned in pyproject.toml)
- `runtime.context` modification for non-serializable tool objects (MCP client connections)
- Sub-graph must be registered as a compiled `StateGraph` node at compile time
- No changes to `mcp` (non-mixed) mode code path
- Frontend error type must accept both `string` and `{type: string, message: string}`
- Existing `messages_from_result()` and `references_from_result()` preserved for compose node
- `ToolCallTransformer` remains in parent graph compile transformers

---

## File Map

| File | Change |
|------|--------|
| `src/rag_agent/graphs/state.py` | Add `MCPSubGraphState` TypedDict |
| `src/rag_agent/graphs/nodes/mixed.py` | Add `call_llm_node()`, `run_tools_node()`, `route()`, `run_mixed_mcp_setup()`, `build_mcp_sub_graph()`. Adapt `run_mixed_compose_node()`. Add `extract_tool_invocations_from_messages()`. |
| `src/rag_agent/graphs/chat_agent.py` | Replace `mixed_mcp` node with `mixed_mcp_setup` + `mcp_sub_graph` sub-graph |
| `frontend/src/lib/types/chat.ts` | Update `MessageReferences.error` union type |
| `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py` | Update tests for new node API |
| `tests/workflow_tests/test_langgraph_chat_agent_modes.py` | Update graph structure assertions |

---

### Task 1: Add sub-graph state and build sub-graph nodes

**Files:**
- Modify: `src/rag_agent/graphs/state.py`
- Modify: `src/rag_agent/graphs/nodes/mixed.py`

**Interfaces:**
- Produces: `MCPSubGraphState` TypedDict, `call_llm_node(state, config, runtime) -> MCPSubGraphState`, `run_tools_node(state, config, runtime) -> MCPSubGraphState`, `route(state) -> Literal["run_tools", "__end__"]`, `build_mcp_sub_graph() -> CompiledStateGraph`

- [ ] **Step 1: Add `MCPSubGraphState` to `state.py`**

Append after `ChatGraphState`:

```python
class MCPSubGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int
```

Add import:
```python
from typing_extensions import NotRequired
```

And add the `MCP_MAX_ROUNDS` constant after the imports at top of `mixed.py`:
```python
MCP_MAX_ROUNDS = 10
```

- [ ] **Step 2: Add sub-graph node functions to `mixed.py`**

Append the following functions before the existing `run_mixed_mcp_node` (which will be replaced in Task 2):

```python
async def call_llm_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    context = get_runtime_context(runtime)
    tools = cast(list | None, context.get("mcp_subgraph_tools"))
    model_id = cast(str | None, context.get("mcp_subgraph_model_id")) or get_llm().model_id
    model = get_llm(model_id=model_id)
    if tools:
        model = model.bind_tools(list(tools))
    remaining = state.get("remaining_steps", MCP_MAX_ROUNDS)
    if remaining <= 0:
        return {"messages": [AIMessage(content="Tool call limit reached.")]}
    response = await model.ainvoke(state["messages"], config=config)
    return {"messages": [response], "remaining_steps": remaining - 1}


async def run_tools_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    context = get_runtime_context(runtime)
    tools = cast(list | None, context.get("mcp_subgraph_tools"))
    if not tools:
        return {"messages": []}
    tool_node = ToolNode(list(tools))
    result = await tool_node.ainvoke({"messages": [state["messages"][-1]]}, config=config)
    return {"messages": cast(list, result.get("messages", []))}


def route(state: MCPSubGraphState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "run_tools"
    return "__end__"
```

Add imports at top of `mixed.py`:
```python
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.rag_agent.graphs.state import MCPSubGraphState
```

- [ ] **Step 3: Add `build_mcp_sub_graph()` factory function**

```python
def build_mcp_sub_graph() -> CompiledStateGraph:
    sub_graph = StateGraph(MCPSubGraphState, context_schema=ChatGraphContext)
    sub_graph.add_node("call_llm", call_llm_node)
    sub_graph.add_node("run_tools", run_tools_node)
    sub_graph.add_conditional_edges("call_llm", route, {"run_tools": "run_tools", "__end__": END})
    sub_graph.add_edge("run_tools", "call_llm")
    sub_graph.set_entry_point("call_llm")
    return sub_graph.compile()
```

- [ ] **Step 4: Add `extract_tool_invocations_from_messages()` helper**

Adapted from `mcp_agent_executor._extract_tool_invocations()`:

```python
def extract_tool_invocations_from_messages(messages: list[object]) -> list[dict[str, object]]:
    pending: dict[str, dict[str, object]] = {}
    invocations: list[dict[str, object]] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in cast(list[dict], msg.tool_calls or []):
                tc_id = str(tc.get("id", "") or "")
                name = str(tc.get("name", "") or "")
                if name:
                    pending[tc_id] = {"tool_name": name, "args": tc.get("args", {})}
            continue
        if isinstance(msg, ToolMessage):
            tc_id = str(getattr(msg, "tool_call_id", "") or "")
            content = str(getattr(msg, "content", "") or "")
            name = str(getattr(msg, "name", "") or "")
            error_text = content if getattr(msg, "status", "") == "error" else None
            rec = pending.pop(tc_id, {"tool_name": name, "args": None})
            rec["result"] = content
            if error_text:
                rec["error"] = error_text
            invocations.append(rec)
            continue
    return invocations
```

- [ ] **Step 5: Verify sub-graph builds**

Run:
```bash
uv run python -c "
from src.rag_agent.graphs.nodes.mixed import build_mcp_sub_graph
g = build_mcp_sub_graph()
print('sub-graph nodes:', list(g.get_graph().nodes.keys()))
print('sub-graph edges:', list(g.get_graph().edges.keys()))
"
```
Expected output:
```
sub-graph nodes: ['__start__', 'call_llm', 'run_tools', '__end__']
sub-graph edges: [('__start__', 'call_llm'), ('call_llm', 'run_tools'), ('call_llm', '__end__'), ('run_tools', 'call_llm')]
```

- [ ] **Step 6: Run existing unit tests to confirm no regression**

Run: `uv run pytest tests/unit_tests/ -x -q`
Expected: All existing tests pass (the new code is not yet wired into the graph, so no existing tests break).

- [ ] **Step 7: Commit**

```bash
git add src/rag_agent/graphs/state.py src/rag_agent/graphs/nodes/mixed.py
git commit -m "feat: add MCP sub-graph state schema and node functions"
```

---

### Task 2: Wire sub-graph into parent graph and adapt compose node

**Files:**
- Modify: `src/rag_agent/graphs/nodes/mixed.py` (add setup node, adapt compose)
- Modify: `src/rag_agent/graphs/chat_agent.py`

**Interfaces:**
- Consumes: `build_mcp_sub_graph()`, `call_llm_node`, `run_tools_node`, `route` (from Task 1)
- Produces: `run_mixed_mcp_setup(state, config, runtime) -> ChatGraphState`, updated `run_mixed_compose_node`

- [ ] **Step 1: Add `run_mixed_mcp_setup()` node to `mixed.py`**

```python
async def run_mixed_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = state.get("messages", [])
    question = latest_user_message(messages)
    chat_history = chat_history_before_latest_user(messages)
    retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name=cast(str | None, context.get("collection_name")),
        filter_docs=rag_runtime.filter_retrieved_docs,
    )
    resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    run_cfg = build_run_config(
        parent_config=config,
        thread_id=thread_id,
        mode="mixed",
        model_id=resolved_model_id,
        session_id=cast(str | None, context.get("session_id")),
        enable_tracing=cast(bool | None, context.get("enable_tracing")),
        mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
    )
    mcp_tools = await load_adapter_tools(
        server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        run_config=run_cfg,
    )
    agent_tools = [retrieval_tool, *mcp_tools] if retrieval_tool else list(mcp_tools)
    system_prompt_text = _build_system_prompt_tools(question, agent_tools)
    input_messages: list[BaseMessage] = []
    for item in chat_history or []:
        converted = _message_to_langchain(item)
        if converted is not None:
            input_messages.append(converted)
    input_messages.append(HumanMessage(content=question))

    runtime.context["mcp_subgraph_tools"] = agent_tools
    runtime.context["mcp_subgraph_model_id"] = resolved_model_id
    runtime.context["mcp_subgraph_question"] = question
    runtime.context["mcp_subgraph_run_cfg"] = run_cfg

    return {
        "messages": [SystemMessage(content=system_prompt_text), *input_messages],
        "progress": "Planning collection and tool search…",
    }
```

The sub-graph's `remaining_steps` starts at the default (MCP_MAX_ROUNDS) via `state.get("remaining_steps", MCP_MAX_ROUNDS)` in `call_llm_node`.
```

Add import:
```python
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
```

Also add the helper function `_build_system_prompt_tools`:
```python
def _build_system_prompt_tools(question: str, tools: list[object]) -> str:
    from src.rag_agent.prompts.mcp_agent_prompts import SYSTEM_PROMPT_MIXED, TOOL_SUMMARY_PLACEHOLDER
    from src.rag_agent.infrastructure.mcp_agent_executor import _build_tool_summary
    return SYSTEM_PROMPT_MIXED.replace(TOOL_SUMMARY_PLACEHOLDER, _build_tool_summary(tools))
```

And `_message_to_langchain`:
```python
def _message_to_langchain(m: object) -> BaseMessage | None:
    if m is None:
        return None
    if isinstance(m, Mapping):
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")
        if not content:
            return None
        if role in {"assistant", "ai"}:
            return AIMessage(content=content)
        return HumanMessage(content=content)
    msg_type = str(getattr(m, "type", "") or getattr(m, "role", "") or "").strip().lower()
    content = str(getattr(m, "content", "") or "")
    if not content:
        return None
    if msg_type in {"assistant", "ai"}:
        return AIMessage(content=content)
    return HumanMessage(content=content)
```

Add imports:
```python
from collections.abc import Mapping
```

- [ ] **Step 2: Adapt `run_mixed_compose_node`**

Replace the existing `run_mixed_compose_node`:

```python
async def run_mixed_compose_node(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    if not messages:
        result = {"final_answer": "Mixed-mode execution did not produce a result."}
        return {"messages": messages_from_result("mixed", result, []), "references": {}}

    context = get_runtime_context(_runtime) if _runtime else {}
    question = context.get("mcp_subgraph_question") or latest_user_message(messages) or ""
    tool_invocations = extract_tool_invocations_from_messages(messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    final_answer = _latest_agent_final_answer(messages) or ""

    retrieval_state = None
    raw_tools = context.get("mcp_subgraph_tools")
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            if hasattr(tool, "_retrieval_state"):
                retrieval_state = getattr(tool, "_retrieval_state", None)
                break
    retrieval_docs = (
        cast(list[object], retrieval_state.get("docs", []))
        if isinstance(retrieval_state, dict)
        else []
    )

    workflow_policy = workflow_policy_for_request(mode="mixed", question=question)
    policy_applied, missing_capabilities, policy_failure_message = enforce_workflow_policy(
        policy=workflow_policy,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    )
    policy_error = policy_failure_message if policy_applied and missing_capabilities else None
    if policy_error:
        final_answer = policy_error
    tool_failure_error = tool_failure_summary(cast(list[dict[str, object]], tool_invocations))
    if not policy_error and is_trivial_answer(final_answer) and tool_failure_error:
        final_answer = tool_failure_error
        policy_error = tool_failure_error
    if retrieval_docs and question:
        retrieval_docs = rag_runtime.rerank_retrieved_docs(
            question,
            cast(list[Any], retrieval_docs),
            enable_reranker=cast(bool | None, context.get("enable_reranker")),
        )
    retrieval_error = oracle_retrieval_error(
        retrieval_state=retrieval_state,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    )
    if not policy_error and retrieval_error:
        final_answer = ORACLE_RETRIEVAL_FAILED_ANSWER
        policy_error = ORACLE_RETRIEVAL_FAILED_ANSWER
    if not policy_error and oracle_retrieval_used_without_context(
        retrieval_state=retrieval_state,
        retrieval_docs=cast(list[Any], retrieval_docs),
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    ):
        final_answer = NO_ORACLE_CONTEXT_ANSWER

    result: dict[str, object] = {
        "final_answer": final_answer,
        "error": policy_error,
        "outcome": "error" if policy_error else "success",
        "standalone_question": question or None,
        "citations": rag_runtime.citations_from_docs(cast(list[Any], retrieval_docs)),
        "reranker_docs": rag_runtime.serialize_docs(cast(list[Any], retrieval_docs)),
        "context_usage": {"retrieved_docs_count": len(retrieval_docs)}
        if retrieval_docs
        else None,
        "mcp_used": bool(tools_used),
        "mcp_tools_used": tools_used,
        "mcp_tool_invocations": tool_invocations,
    }
    messages_out = messages_from_result("mixed", result, messages)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": messages_out,
        "references": references,
    }
```

- [ ] **Step 3: Wire sub-graph in `chat_agent.py`**

Replace the `mixed_mcp` node and its edges with the setup + sub-graph:

```python
# Replace:
# graph.add_node("mixed_mcp", run_mixed_mcp_node)
# with:
graph.add_node("mixed_mcp_setup", run_mixed_mcp_setup)

# Add the sub-graph as a compiled node
from src.rag_agent.graphs.nodes.mixed import build_mcp_sub_graph
mcp_sub_graph = build_mcp_sub_graph()
graph.add_node("mcp_sub_graph", mcp_sub_graph)

# Replace edges:
# graph.add_edge("mixed_mcp", "mixed_compose")
# with:
graph.add_edge("mixed_mcp_setup", "mcp_sub_graph")
graph.add_edge("mcp_sub_graph", "mixed_compose")
```

The import `build_mcp_sub_graph` should be at the top of `chat_agent.py`.

- [ ] **Step 4: Run workflow tests**

Run: `uv run pytest tests/workflow_tests/test_langgraph_chat_agent_modes.py -x -v`
Expected: Tests may need minor adjustments for new graph structure assertions (e.g., node name checks).

- [ ] **Step 5: Run all unit tests**

Run: `uv run pytest tests/unit_tests/ -x -q`
Expected: All pass or failures are only in tests that need Task 3 updates.

- [ ] **Step 6: Commit**

```bash
git add src/rag_agent/graphs/nodes/mixed.py src/rag_agent/graphs/chat_agent.py
git commit -m "feat: wire MCP sub-graph into mixed mode graph"
```

---

### Task 3: Update backend tests for new sub-graph API

**Files:**
- Modify: `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`
- Modify: `tests/workflow_tests/test_langgraph_chat_agent_modes.py`
- (possible) `tests/workflow_tests/test_langgraph_chat_contract.py`

**Interfaces:**
- Consumes: `build_mcp_sub_graph()`, `extract_tool_invocations_from_messages()` (Task 1), `run_mixed_mcp_setup()`, adapted `run_mixed_compose_node()` (Task 2)

- [ ] **Step 1: Read existing test file**

Read `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py` to understand current test patterns.

- [ ] **Step 2: Update `test_mixed_nodes_use_runtime_context`**

In `tests/workflow_tests/test_langgraph_chat_agent_modes.py`, update the test to replace `mixed_mcp` node checks with `mixed_mcp_setup` + `mcp_sub_graph` node checks.

- [ ] **Step 3: Update `test_langgraph_mcp_mixed_nodes.py`**

Replace direct `run_mixed_mcp_node` calls with tests that exercise the sub-graph + compose flow. Key test scenarios:
- Sub-graph route returns `"run_tools"` when last message has tool_calls
- Sub-graph route returns `"__end__"` when last message has no tool_calls
- `extract_tool_invocations_from_messages` pairs AIMessage.tool_calls with ToolMessages
- `run_mixed_compose_node` produces correct references from sub-graph output messages

- [ ] **Step 4: Run updated tests**

Run: `uv run pytest tests/unit_tests/test_langgraph_mcp_mixed_nodes.py tests/workflow_tests/test_langgraph_chat_agent_modes.py -x -v`
Expected: All pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/unit_tests/ tests/workflow_tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: update tests for MCP sub-graph architecture"
```

---

### Task 4: Update frontend error contract type

**Files:**
- Modify: `frontend/src/lib/types/chat.ts`

- [ ] **Step 1: Read current type definition**

```bash
cat frontend/src/lib/types/chat.ts | grep -n "error"
```

- [ ] **Step 2: Update `MessageReferences.error` union type**

Change:
```typescript
error?: string;
```
To:
```typescript
error?: string | { type: string; message: string };
```

- [ ] **Step 3: Verify frontend builds**

Run: `cd frontend && pnpm build`
Expected: No type errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/types/chat.ts
git commit -m "feat: update frontend error type for structured backend errors"
```

---

### Task 5: Integration verification

**Files:** None (manual test)

- [ ] **Step 1: Start the Agent Server**

Run: `uv run langgraph dev` (or `make up` if compose is configured)
Wait for: `Agent Server running at http://127.0.0.1:2024`

- [ ] **Step 2: Submit a mixed-mode request with MCP tools**

```bash
curl -s -X POST http://127.0.0.1:2024/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "chat_agent",
    "input": {"messages": [{"role": "user", "content": "Calculate 42 * 7 using the calculator tool"}]},
    "config": {"configurable": {"mode": "mixed"}}
  }' 2>/dev/null | grep -o '"type":"[^"]*"' | sort | uniq -c
```

Expected: Output includes `tools` and `messages` event types, with tool events preceding the final answer.

- [ ] **Step 3: Verify tool events in stream**

Run the same request and look for:
- `tool-started` events for each tool call
- `tool-finished` events with results
- Token-level `content-block-delta` events for LLM output

- [ ] **Step 4: Verify thread replay works**

Get the thread ID from step 2, then:
```bash
curl -s http://127.0.0.1:2024/threads/<thread_id>/state
```
Expected: Final `AIMessage` has correct `additional_kwargs` with `citations`, `mcp_tool_invocations`, etc.

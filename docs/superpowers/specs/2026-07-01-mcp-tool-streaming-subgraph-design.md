# MCP Mixed-Mode Tool Streaming Sub-graph Design

## Goal

Replace the current single-node MCP tool loop in mixed mode with a proper LangGraph sub-graph so that tool calls and LLM output stream in real time to the frontend via the `tools` and `messages` SSE channels. Align the frontend/backend contract to a single canonical field set.

## Context

Mixed mode runs retrieval + MCP tools in one request. The current architecture packs the entire MCP agent turn (model calls, tool invocations, retries) into one LangGraph node (`mixed_mcp`). LLM streaming is suppressed via `suppress_llm_streaming()`. Tool events appear only after the node completes, batched into `AIMessage.additional_kwargs`. The frontend receives no incremental tool progress.

The original goal of `suppress_llm_streaming` was to prevent conflated output from nested agent loops. LangGraph 1.2.6 now supports proper sub-graph streaming where nested graph events propagate to the parent stream through `DuplexStream`.

## Current architecture

```text
bootstrap → mixed_route → mixed_retrieval → mixed_mcp → mixed_compose → END

mixed_mcp (single async function node):
  ├── load_adapter_tools()                 ← MCP connections
  ├── suppress_llm_streaming(model)        ← kills token/tool events
  ├── run_mcp_agent_turn():
  │    ├── _run_graph_native_tool_loop()   ← or create_agent() + middleware
  │    │    ├── model.ainvoke()            ← nostream
  │    │    ├── ToolNode.ainvoke()         ← no separate superstep
  │    │    └── repeat...
  │    └── returns MCPAnswerExecutionResult
  ├── policy enforcement
  └── returns {"mixed_result": {...}, "mixed_state_messages": [...]}
```

Streaming events emitted: one `values` superstep with the final batch.

## Proposed architecture

### Graph layout

```text
bootstrap → mixed_route → mixed_retrieval → mixed_mcp_setup → mcp_sub_graph → mixed_compose → END
                                                                    │
                                                             ┌──────┴──────┐
                                                             │  call_llm   │
                                                             │  route      │
                                                             │  run_tools  │
                                                             └──────┬──────┘
                                                                    │ (conditional loop)
```

### Node responsibilities

| Node | Type | What it does |
|------|------|--------------|
| `mixed_mcp_setup` | Regular (async fn) | Loads MCP tools via `load_adapter_tools()`, builds retrieval tool, stores both in `runtime.context["mcp_subgraph_tools"]`. Sets `progress` context string. Returns sub-graph input state. |
| `mcp_sub_graph` | `CompiledStateGraph` | LangGraph sub-graph with its own state schema. Contains `call_llm`, `run_tools`, `route` nodes. Registered at compile time in `build_chat_agent()`. Tools resolved from `runtime.context` at runtime. |
| `mixed_compose` | Regular (async fn) | Unchanged. Reads sub-graph output messages from parent state, applies policy enforcement, produces final `AIMessage` with references. |

### Sub-graph state schema

```python
class MCPSubGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
```

The sub-graph does not carry its own tool list in state. Tools are non-serializable (MCP client connections) and are resolved from `runtime.context` inside each node.

### Sub-graph nodes

**`call_llm(node)`** — reads `runtime.context["mcp_subgraph_tools"]`, binds them to the LLM, invokes, returns:

```python
{"messages": [AIMessage(tool_calls=[...])]}  # or content if no tools
```

No `suppress_llm_streaming`. The LangGraph runtime emits `messages`-channel token events for this call naturally.

**`run_tools(node)`** — reads `runtime.context["mcp_subgraph_tools"]`, creates an ephemeral `ToolNode` from them, calls `ToolNode.ainvoke()` on the last message's tool calls:

```python
{"messages": [ToolMessage(...), ...]}  # one per tool call
```

LangGraph runtime detects the `AIMessage.tool_calls` → `ToolMessage` transition and emits `tools`-channel events (`tool-started` → `tool-output-delta` → `tool-finished`).

**`route(edge)`** — conditional edge:

```python
def route(state: MCPSubGraphState) -> Literal["run_tools", END]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "run_tools"
    return END
```

After `run_tools`, the edge unconditionally goes back to `call_llm` (LangGraph handles this as the edge definition in `build_chat_agent`).

### Sub-graph compilation

```python
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

mcp_sub_graph = StateGraph(MCPSubGraphState, context_schema=ChatGraphContext)
mcp_sub_graph.add_node("call_llm", call_llm_node)
mcp_sub_graph.add_node("run_tools", run_tools_node)
mcp_sub_graph.add_conditional_edges("call_llm", route, {"run_tools": "run_tools", END: END})
mcp_sub_graph.add_edge("run_tools", "call_llm")
mcp_sub_graph.set_entry_point("call_llm")
compiled_sub_graph = mcp_sub_graph.compile()
```

### Parent graph integration

In `build_chat_agent()`:

```python
graph.add_node("mixed_mcp_setup", run_mixed_mcp_setup)  # new
graph.add_node("mcp_sub_graph", compiled_sub_graph)       # replaces mixed_mcp
graph.add_node("mixed_compose", run_mixed_compose_node)

graph.add_edge("mixed_retrieval", "mixed_mcp_setup")
graph.add_edge("mixed_mcp_setup", "mcp_sub_graph")
graph.add_edge("mcp_sub_graph", "mixed_compose")
```

### `mixed_mcp_setup` node

```python
async def run_mixed_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = state["messages"]
    question = latest_user_message(messages)
    chat_history = chat_history_before_latest_user(messages)

    # Load tools (same as current mixed_mcp preamble)
    retrieval_tool = rag_runtime.build_oracle_retrieval_tool(...)
    mcp_tools = await load_adapter_tools(...)
    agent_tools = [retrieval_tool, *mcp_tools]

    resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    run_cfg = build_run_config(..., mode="mixed", ...)

    # Store tools in runtime context for sub-graph nodes
    runtime.context["mcp_subgraph_tools"] = agent_tools
    runtime.context["mcp_subgraph_question"] = question
    runtime.context["mcp_subgraph_chat_history"] = chat_history
    runtime.context["mcp_subgraph_model_id"] = resolved_model_id
    runtime.context["mcp_subgraph_run_cfg"] = run_cfg

    # Build input messages for sub-graph
    system_prompt = _build_system_prompt(question, agent_tools, run_cfg)
    input_messages = [SystemMessage(content=system_prompt), *chat_history, HumanMessage(content=question)]

    return {"messages": input_messages, "progress": "Running tool search…"}
```

### `call_llm` and `run_tools` node signatures

```python
async def call_llm_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    tools = runtime.context.get("mcp_subgraph_tools", [])
    model_id = runtime.context.get("mcp_subgraph_model_id")
    model = get_llm(model_id=model_id)
    if tools:
        model = model.bind_tools(list(tools))
    response = await model.ainvoke(state["messages"], config=config)
    return {"messages": [response]}


async def run_tools_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    tools = runtime.context.get("mcp_subgraph_tools", [])
    tool_node = ToolNode(list(tools))
    result = await tool_node.ainvoke({"messages": [state["messages"][-1]]}, config=config)
    return {"messages": result["messages"]}
```

### `mixed_compose` adaptations

The `mixed_compose` node currently reads `state["mixed_result"]` and `state["mixed_state_messages"]`. After the sub-graph, these keys no longer exist. Instead, `mixed_compose` reads the sub-graph's output messages from `state["messages"]` and extracts tool invocations by scanning the messages (as `_extract_tool_invocations()` currently does):

```python
async def run_mixed_compose_node(state, ...) -> ChatGraphState:
    messages = state["messages"]
    # Extract tool invocations from message history
    tool_invocations = extract_tool_invocations_from_messages(messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    # Find final answer (last AIMessage without tool_calls)
    final_answer = latest_agent_final_answer(messages)
    # Policy enforcement (unchanged)
    # ...
    result = {
        "citations": rag_runtime.citations_from_docs(retrieval_docs),
        "reranker_docs": ...,
        "mcp_used": bool(tools_used),
        "mcp_tools_used": tools_used,
        "mcp_tool_invocations": tool_invocations,
        "standalone_question": question,
        "context_usage": ...,
    }
    # ... produce final AIMessage with references
    messages_out = messages_from_result("mixed", result, messages)
    return {"messages": messages_out, "references": references}
```

Key: the sub-graph's output messages (AIMessages, ToolMessages) are preserved through `messages_from_result()`, which attaches reference metadata to the last AIMessage.

## Streaming behavior changes

| Event channel | Before (single node) | After (sub-graph) |
|---|---|---|
| `messages` | One complete `AIMessage` at end | Token-level `content-block-delta` events per LLM call, plus final `AIMessage` |
| `tools` | None (no per-superstep tool detection) | `tool-started` → `tool-output-delta` → `tool-finished` per tool invocation |
| `values` | One superstep with `{"messages": [...], "progress": ...}` | N supersteps: one per LLM call, one per tool execution, each with `messages` and `progress` |

### Frontend impact

The frontend currently handles tool calls through two paths:
1. `stream.toolCalls` (live `tools` channel events from SDK)
2. `toolCallsFromMessages()` (fallback from parsing `AIMessage.tool_calls`)

With the sub-graph, `stream.toolCalls` becomes the primary path and works reliably for mixed mode — each tool call gets its own `tool-started`/`tool-finished` event. The `toolCallsFromMessages()` fallback still works but is no longer needed for correctness during streaming (it serves as a replay fallback).

The frontend's `".content"` suppression for tool-call-only assistant messages (`content.trim() === "." ? "" : content`) remains because `create_react_agent` and the custom sub-graph both produce `content="."` or empty content for tool-call-only messages. This is a LangChain convention.

## Contract alignment

The `final_answer` key in result dicts is eliminated. The answer flows naturally as `AIMessage.content`. References (citations, tool invocations, context usage) continue to live in `AIMessage.additional_kwargs` and `response_metadata`.

### Fields after change

| Field | Source | Always present? |
|---|---|---|
| `AIMessage.content` | LLM output (the answer) | Yes |
| `additional_kwargs.mode` | `"mixed"` | Yes |
| `additional_kwargs.citations` | Citation normalization | Yes (empty list if none) |
| `additional_kwargs.reranker_docs` | Reranker output | Yes (empty list if none) |
| `additional_kwargs.mcp_tools_used` | Sub-graph extraction | Yes (empty list if none) |
| `additional_kwargs.mcp_tool_invocations` | Sub-graph extraction | Yes (empty list if none) |
| `additional_kwargs.standalone_question` | From setup | If non-null |
| `additional_kwargs.context_usage` | From retrieval | If docs retrieved |
| `additional_kwargs.trace_id` | Langfuse | If tracing enabled |
| `additional_kwargs.error` | Error handling | If error occurred |
| `additional_kwargs.outcome` | Policy result | If set by compose |

### Frontend contract impact

Minimal. The frontend already reads:
- `AIMessage.content` for answer text — unchanged
- `additional_kwargs` for references — unchanged
- `stream.toolCalls` for live tool progress — now works reliably for mixed mode

The `error` field shape is standardized: backend sends `{"type": str, "message": str}` (dict), frontend currently expects `string`. The frontend's `MessageReferences.error` type should be updated to accept both `string | {type: string, message: string}`.

## Files changed

| File | Change |
|---|---|
| `src/rag_agent/graphs/state.py` | Add `MCPSubGraphState` schema |
| `src/rag_agent/graphs/chat_agent.py` | Replace `mixed_mcp` node with `mixed_mcp_setup` + `mcp_sub_graph` sub-graph |
| `src/rag_agent/graphs/nodes/mixed.py` | Add `run_mixed_mcp_setup()`, `call_llm_node()`, `run_tools_node()`, `route()`. Adapt `run_mixed_compose_node()` to read from parent state messages directly. |
| `src/rag_agent/infrastructure/mcp_agent_executor.py` | Remove `suppress_llm_streaming()` import and call. Remove `_run_graph_native_tool_loop()` and managed agent paths (no longer called). |
| `src/rag_agent/runtime/mcp_turn.py` | Remove `run_mcp_agent_turn()` or simplify to just tool loading (called by setup node). |
| `src/rag_agent/infrastructure/mcp_agent.py` | Remove unused wrapper functions if no longer referenced. |
| `frontend/src/lib/types/chat.ts` | Update `MessageReferences.error` to accept `string | {type: string, message: string}` |

## Removed code paths

- `suppress_llm_streaming()` — no longer called anywhere
- `_run_graph_native_tool_loop()` — replaced by sub-graph nodes
- `create_agent()` + middleware (`OCIToolCallContentMiddleware`, `LLMToolSelectorMiddleware`, `ToolRetryMiddleware`, `ToolCallLimitMiddleware`) — no longer used; sub-graph handles the loop natively
- `get_mcp_answer_execution_with_langchain_agent_async()` — the entire function is replaced; setup + sub-graph invoke replace it
- `_extract_answer_and_tools()` — replaced by message scanning in `mixed_compose`

## Retry and policy handling

### `require_tool_call` retry
The current retry logic (lines 580-637 of `mcp_agent_executor.py`) adds a "call a tool" message and re-runs when no tools were invoked. In the sub-graph approach, this can be:
- Handled in `mixed_compose`: if the sub-graph output has no tool calls and `require_tool_call` is enabled, re-enter the sub-graph with an appended HumanMessage
- OR handled as a sub-graph-level loop with a counter

Recommendation: handle at the sub-graph entry point. The `mixed_mcp_setup` node checks whether this is a first attempt or a retry, and adjusts input messages accordingly. A counter in `runtime.context["mcp_subgraph_retry_count"]` prevents infinite loops.

### Policy enforcement
Unchanged. `mixed_compose` still runs `enforce_workflow_policy()`, `tool_failure_summary()`, and `is_trivial_answer()` on the sub-graph's output messages and extracted tool invocations.

### Tool call limit
The sub-graph's `run_tools` node naturally limits tool calls per superstep (one `AIMessage.tool_calls` per LLM call). The total round limit can be enforced via a counter in `runtime.context` or by adding `remaining_steps` to the sub-graph state.

## Migration plan

1. Add `MCPSubGraphState` to `state.py`
2. Build sub-graph nodes (`call_llm`, `run_tools`, `route`) in `mixed.py`
3. Add `run_mixed_mcp_setup` in `mixed.py`
4. Adapt `run_mixed_compose_node` to read from parent state messages
5. Wire sub-graph into `chat_agent.py`, replacing `mixed_mcp`
6. Remove `suppress_llm_streaming` call from `mcp_agent_executor.py`
7. Remove unused functions from `mcp_agent_executor.py`, `mcp_agent.py`, `mcp_turn.py`
8. Update frontend `MessageReferences.error` type
9. Test with a real mixed-mode request (RAG + MCP tool call) and verify:
   - `stream.toolCalls` shows live tool progress
   - Token events arrive for each LLM call
   - Final answer has correct references
   - Replay works after thread state fetch

## Out of scope

- Changing the MCP server connection or tool-loading mechanism
- Replacing the `rag` or `mcp` (non-mixed) node patterns
- Frontend UI changes (tool rendering already works)
- Frontend `useStream` provider changes
- Altering the `direct` or `rag` graph modes

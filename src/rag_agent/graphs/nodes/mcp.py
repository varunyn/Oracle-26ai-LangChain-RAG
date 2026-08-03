from __future__ import annotations

from typing import cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import is_trivial_answer
from src.rag_agent.graphs.nodes.mixed import (
    _latest_agent_final_answer,
    extract_tool_invocations_from_messages,
)
from src.rag_agent.graphs.nodes.references import messages_from_result
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.graphs.tool_agent_turn import (
    get_tool_agent_turn,
    prepare_tool_agent_turn,
)
from src.rag_agent.runtime.mcp_turn import tool_failure_summary


async def run_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    turn = await prepare_tool_agent_turn(
        state=state,
        parent_config=config,
        runtime=runtime,
        mode="mcp",
    )
    runtime.context["tool_agent_turn"] = turn

    return {
        "messages": [],
    }


async def run_mcp_compose(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    if not messages:
        result = {"final_answer": "MCP execution did not produce a result."}
        return {"messages": messages_from_result("mcp", result, []), "references": {}}

    turn = get_tool_agent_turn(_runtime) if _runtime else None
    tool_invocations = extract_tool_invocations_from_messages(messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    final_answer = _latest_agent_final_answer(messages) or ""

    tool_failure_error = tool_failure_summary(cast(list[dict[str, object]], tool_invocations))
    if is_trivial_answer(final_answer) and tool_failure_error:
        final_answer = tool_failure_error

    result: dict[str, object] = {
        "final_answer": final_answer,
        "error": None,
        "standalone_question": turn["question"] if turn else None,
        "citations": [],
        "reranker_docs": [],
        "context_usage": None,
        "mcp_used": bool(tools_used),
        "mcp_tools_used": tools_used,
        "mcp_tool_invocations": tool_invocations,
    }
    messages_out = messages_from_result("mcp", result, messages)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": messages_out,
        "references": references,
    }

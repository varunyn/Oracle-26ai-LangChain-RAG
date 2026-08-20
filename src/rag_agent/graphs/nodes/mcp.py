from __future__ import annotations

import logging
from typing import cast

from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import is_trivial_answer
from src.rag_agent.graphs.nodes.references import messages_from_result
from src.rag_agent.graphs.runtime import stable_terminal_message_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.graphs.tool_agent_execution import (
    analyze_tool_execution,
    messages_since_latest_user,
)
from src.rag_agent.graphs.tool_agent_turn import (
    ToolAgentTurn,
    mark_tool_agent_turn_terminal,
    prepare_tool_agent_turn,
    reconstruct_tool_agent_turn,
    release_tool_agent_turn,
    release_tool_agent_turn_after_failure,
)
from src.rag_agent.runtime.mcp_turn import tool_failure_summary

logger = logging.getLogger(__name__)


async def run_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    turn = await prepare_tool_agent_turn(
        state=state, parent_config=config, runtime=runtime, mode="mcp"
    )
    await release_tool_agent_turn(config, turn)
    return {
        "messages": [],
    }


async def run_mcp_compose(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    turn = None
    if messages and _runtime and _config:
        turn = await reconstruct_tool_agent_turn(
            state=state, parent_config=_config, runtime=_runtime, mode="mcp"
        )
    try:
        result = await _compose_mcp_result(state, turn)
    except BaseException:
        if turn and _config:
            await _release_after_failure(_config, turn)
        raise
    if turn and _config:
        try:
            await _mark_terminal(_config, turn)
            await release_tool_agent_turn(_config, turn)
        except BaseException:
            await _release_after_failure(_config, turn)
            raise
    return result


async def _compose_mcp_result(
    state: ChatGraphState, turn: ToolAgentTurn | None = None
) -> ChatGraphState:
    messages = state.get("messages", [])
    message_id = _terminal_message_id("mcp", turn)
    if not messages:
        empty_result: dict[str, object] = {
            "final_answer": "MCP execution did not produce a result."
        }
        return {
            "messages": cast(
                list[AnyMessage],
                messages_from_result("mcp", empty_result, [], message_id=message_id),
            ),
            "references": {},
        }

    transcript = analyze_tool_execution(messages_since_latest_user(cast(list[object], messages)))
    tool_invocations = transcript["tool_invocations"]
    tools_used = transcript["tools_used"]
    final_answer = transcript["final_answer"]

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
    messages_out = messages_from_result("mcp", result, messages, message_id=message_id)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": cast(list[AnyMessage], messages_out),
        "references": references,
    }


def _terminal_message_id(mode: str, turn: ToolAgentTurn | None) -> str | None:
    lease = turn.get("lease") if turn else None
    thread_id = getattr(lease, "thread_id", None)
    turn_id = getattr(lease, "turn_id", None)
    if isinstance(thread_id, str) and isinstance(turn_id, str):
        return cast(str, stable_terminal_message_id(mode, thread_id, turn_id))
    return None


async def _mark_terminal(config: RunnableConfig, turn: ToolAgentTurn) -> None:
    message_id = _terminal_message_id("mcp", turn)
    if message_id is None:
        return
    await mark_tool_agent_turn_terminal(config, turn, message_id)


async def _release_after_failure(config: RunnableConfig, turn: ToolAgentTurn) -> None:
    try:
        await release_tool_agent_turn_after_failure(config, turn)
    except BaseException:
        logger.warning("Failed stale tool-agent lease cleanup", exc_info=True)

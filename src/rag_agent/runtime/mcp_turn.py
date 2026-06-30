"""MCP agent turn execution for chat runtime modes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
from src.rag_agent.infrastructure.mcp_agent import (
    get_mcp_answer_async,
    get_mcp_answer_result_async,
)
from src.rag_agent.workflows.mcp_repeated import run_repeated_mcp_workflow
from src.rag_agent.workflows.workflow_intent import should_use_repeated_workflow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPAgentTurn:
    answer: str
    tools_used: list[str]
    tool_invocations: list[dict[str, object]]
    resolved_model_id: str
    state_messages: list[object]


def question_explicitly_references_mcp_tools(
    question: str,
    mcp_tools: list[BaseTool],
) -> bool:
    lower_question = question.strip().lower()
    if not lower_question:
        return False
    for tool in mcp_tools:
        tool_name = str(getattr(tool, "name", "") or "").strip().lower()
        if not tool_name:
            continue
        if tool_name in lower_question:
            return True
        humanized = tool_name.replace("_", " ")
        if humanized in lower_question:
            return True
    return False


def tool_failure_summary(tool_invocations: list[dict[str, object]]) -> str | None:
    failed_tools: list[str] = []
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool_name") or "").strip()
        error_text = str(invocation.get("error") or "").strip()
        if error_text:
            if tool_name and tool_name not in failed_tools:
                failed_tools.append(tool_name)
    if not failed_tools:
        return None
    joined = ", ".join(failed_tools)
    return f"Workflow failed because tool execution failed: {joined}. See tool output for details."


async def run_mcp_agent_turn(
    *,
    question: str,
    chat_history: list[object],
    resolved_model_id: str,
    run_config: RunnableConfig,
    mode: str,
    mcp_server_keys: list[str] | None,
    require_tool_call: bool,
    repeated_workflow_enabled: bool,
    workflow_checkpoint_path: str | None,
    extra_tools: list[BaseTool] | None = None,
    require_mcp_tool_call_when_referenced: bool = False,
) -> MCPAgentTurn:
    tool_load_started = time.perf_counter()
    mcp_tools = await load_adapter_tools(
        server_keys=mcp_server_keys,
        run_config=run_config,
    )
    logger.info(
        "chat_runtime_mcp_tools_loaded mode=%s tool_count=%d elapsed_ms=%.1f",
        mode,
        len(mcp_tools),
        (time.perf_counter() - tool_load_started) * 1000,
    )
    agent_tools = [*(extra_tools or []), *mcp_tools]
    explicit_mcp_required = (
        require_mcp_tool_call_when_referenced
        and question_explicitly_references_mcp_tools(question, mcp_tools)
    )
    effective_require_tool_call = require_tool_call or explicit_mcp_required
    repeated_result = None
    repeated_workflow_selected = repeated_workflow_enabled and await should_use_repeated_workflow(
        question=question,
        tools=agent_tools,
        model_id=resolved_model_id,
        run_config=run_config,
    )
    if repeated_workflow_selected:
        repeated_result = await run_repeated_mcp_workflow(
            question=question,
            model_id=resolved_model_id,
            tools=agent_tools,
            run_config=run_config,
            require_tool_call=effective_require_tool_call,
            get_answer=get_mcp_answer_async,
            checkpoint_path=workflow_checkpoint_path,
            chat_history=chat_history,
        )
    if repeated_result is None and not repeated_workflow_selected:
        execution_result = await get_mcp_answer_result_async(
            question,
            chat_history=chat_history,
            model_id=resolved_model_id,
            tools=agent_tools,
            run_config=run_config,
            require_tool_call=effective_require_tool_call,
        )
        answer = execution_result.answer
        tools_used = execution_result.tools_used
        tool_invocations = execution_result.tool_invocations
        state_messages = execution_result.state_messages
    elif repeated_result is None:
        answer = (
            "I could not identify a work queue from the discovery tool results, "
            "so I stopped before processing individual work units."
        )
        tools_used = []
        tool_invocations = []
        state_messages = []
    else:
        answer, tools_used, tool_invocations = repeated_result
        state_messages = []
    return MCPAgentTurn(
        answer=answer,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
        resolved_model_id=resolved_model_id,
        state_messages=_normalize_state_messages(state_messages, answer=answer),
    )


def _normalize_state_messages(messages: list[object], *, answer: str) -> list[object]:
    if messages:
        return messages
    if not answer:
        return []
    return [AIMessage(content=answer)]

"""MCP answer orchestration backed by LangChain agent executor."""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ..prompts.mcp_agent_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MIXED
from .async_utils import run_coroutine_sync
from .mcp_adapter_runtime import load_adapter_tools
from .mcp_agent_executor import (
    MCPAnswerExecutionResult,
    get_mcp_answer_execution_with_langchain_agent_async,
    get_mcp_answer_with_langchain_agent_async,
)
from .mcp_settings import get_mcp_settings

logger = logging.getLogger(__name__)

__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_MIXED",
    "get_mcp_answer",
    "get_mcp_answer_async",
    "get_mcp_answer_result_async",
]


async def _get_mcp_answer_impl(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if get_mcp_settings().enable_mcp_tools is False:
        return "", [], []

    resolved_tools = tools
    if resolved_tools is None:
        resolved_tools = await load_adapter_tools(server_keys=server_keys, run_config=run_config)

    if not resolved_tools:
        return "MCP tools are currently unavailable. Please try again.", [], []

    return await get_mcp_answer_with_langchain_agent_async(
        question=question,
        chat_history=chat_history,
        model_id=model_id,
        tools=resolved_tools,
        run_config=run_config,
        require_tool_call=require_tool_call,
    )


async def _get_mcp_answer_result_impl(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> MCPAnswerExecutionResult:
    if get_mcp_settings().enable_mcp_tools is False:
        return MCPAnswerExecutionResult(
            answer="",
            tools_used=[],
            tool_invocations=[],
            state_messages=[],
        )

    resolved_tools = tools
    if resolved_tools is None:
        resolved_tools = await load_adapter_tools(server_keys=server_keys, run_config=run_config)

    if not resolved_tools:
        return MCPAnswerExecutionResult(
            answer="MCP tools are currently unavailable. Please try again.",
            tools_used=[],
            tool_invocations=[],
            state_messages=[],
        )

    return await get_mcp_answer_execution_with_langchain_agent_async(
        question=question,
        chat_history=chat_history,
        model_id=model_id,
        tools=resolved_tools,
        run_config=run_config,
        require_tool_call=require_tool_call,
    )


def get_mcp_answer(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    return cast(
        tuple[str, list[str], list[dict[str, Any]]],
        run_coroutine_sync(
            _get_mcp_answer_impl(
                question,
                chat_history=chat_history,
                model_id=model_id,
                server_keys=server_keys,
                tools=tools,
                require_tool_call=require_tool_call,
                run_config=run_config,
            )
        ),
    )


async def get_mcp_answer_async(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    return await _get_mcp_answer_impl(
        question,
        chat_history=chat_history,
        model_id=model_id,
        server_keys=server_keys,
        tools=tools,
        require_tool_call=require_tool_call,
        run_config=run_config,
    )


async def get_mcp_answer_result_async(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> MCPAnswerExecutionResult:
    return await _get_mcp_answer_result_impl(
        question,
        chat_history=chat_history,
        model_id=model_id,
        server_keys=server_keys,
        tools=tools,
        require_tool_call=require_tool_call,
        run_config=run_config,
    )

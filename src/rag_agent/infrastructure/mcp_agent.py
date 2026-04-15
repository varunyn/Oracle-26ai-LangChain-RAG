"""MCP answer orchestration backed by LangChain agent executor."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ..prompts.mcp_agent_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MIXED
from .direct_mcp_tools import get_mcp_tools_async
from .mcp_agent_executor import get_mcp_answer_with_langchain_agent_async
from .mcp_settings import get_mcp_settings

logger = logging.getLogger(__name__)

__all__ = ["SYSTEM_PROMPT", "SYSTEM_PROMPT_MIXED", "get_mcp_answer", "get_mcp_answer_async"]


def _run_coroutine_in_thread(coro: Coroutine[object, object, object]) -> object:
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("Thread runner did not return a value.")
    return result["value"]


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
        resolved_tools = await get_mcp_tools_async(server_keys=server_keys, run_config=run_config)

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


def get_mcp_answer(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _get_mcp_answer_impl(
                question,
                chat_history=chat_history,
                model_id=model_id,
                server_keys=server_keys,
                tools=tools,
                require_tool_call=require_tool_call,
                run_config=run_config,
            )
        )

    return cast(
        tuple[str, list[str], list[dict[str, Any]]],
        _run_coroutine_in_thread(
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

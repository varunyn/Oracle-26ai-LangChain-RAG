"""
MCP tool loop using the same ChatOCIGenAI as RAG (get_llm).

Simple workshop-style loop: bind tools to LLM, invoke, execute tool_calls, repeat.
Same LLM and config as RAG (answer_generator, reranker). No LangGraph, no create_oci_agent.
OCI Gen AI does not support parallel tool calls; we process one tool call per round.
"""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Coroutine, Mapping, Sequence
from typing import Protocol, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ..prompts.mcp_agent_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_MIXED,
    TOOL_SUMMARY_PLACEHOLDER,
)
from .direct_mcp_tools import get_mcp_tools_async
from .mcp_settings import get_mcp_settings
from .oci_models import get_llm

logger = logging.getLogger(__name__)

__all__ = ["SYSTEM_PROMPT", "SYSTEM_PROMPT_MIXED", "get_mcp_answer", "get_mcp_answer_async"]

_RAW_TOOL_CALL_PATTERN = re.compile(r"^[\w\.]+\s*\([^)]*\)$")


def _looks_like_unresolved_tool_call(text: str) -> bool:
    candidate = text.strip()
    return bool(candidate) and len(candidate) <= 200 and bool(_RAW_TOOL_CALL_PATTERN.fullmatch(candidate))


class _ToolInvoker(Protocol):
    def invoke(self, tool_call: dict[str, object]) -> object: ...

    def ainvoke(self, tool_call: dict[str, object]) -> Coroutine[object, object, object]: ...


class _ToolBound(Protocol):
    def invoke(
        self, messages: Sequence[BaseMessage], *, config: RunnableConfig | None = None
    ) -> AIMessage: ...

    def ainvoke(
        self, messages: Sequence[BaseMessage], *, config: RunnableConfig | None = None
    ) -> Coroutine[object, object, AIMessage]: ...


class _ToolBindable(Protocol):
    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: object) -> _ToolBound: ...


def _message_to_langchain(m: object) -> BaseMessage | None:
    """Convert dict or message to HumanMessage or AIMessage. Skip if no content."""
    if m is None:
        return None
    if isinstance(m, Mapping):
        data = cast(Mapping[str, object], m)
        role_value = data.get("role")
        role = role_value.lower() if isinstance(role_value, str) else ""
        content_value = data.get("content")
        if content_value in (None, ""):
            return None
        content_text = _normalize_message_content(content_value)
        if not content_text:
            return None
        if role in ("human", "user"):
            return HumanMessage(content=content_text)
        if role in ("ai", "assistant"):
            return AIMessage(content=content_text)
        return HumanMessage(content=content_text)

    content_attr = getattr(m, "content", None)
    if content_attr is None:
        return None
    content_text = _normalize_message_content(cast(object, content_attr))
    if not content_text:
        return None
    msg_type_value = getattr(m, "type", None) or getattr(m, "role", None) or ""
    msg_type = msg_type_value.lower() if isinstance(msg_type_value, str) else ""
    if msg_type in ("human", "user"):
        return HumanMessage(content=content_text)
    if msg_type in ("ai", "assistant"):
        return AIMessage(content=content_text)
    return HumanMessage(content=content_text)


# Cap history so the model answers the current question; keep enough for follow-ups (e.g. OCI namespace then list compartments).
# -1 = no cap; otherwise only the last N messages are sent. 6 = last 3 user/assistant exchanges for multi-step OCI.
MCP_MAX_HISTORY_MESSAGES = 15


def _normalize_message_content(content: object) -> str:
    if isinstance(content, list):
        items = cast(list[object], content)
        parts: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                item_map = cast(Mapping[str, object], item)
                text_value = item_map.get("text")
                parts.append(text_value if isinstance(text_value, str) else "")
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()


def _build_tool_summary(tools: Sequence[BaseTool]) -> str:
    if not tools:
        return "(No tools registered.)"

    lines: list[str] = []
    for tool in tools:
        description = (tool.description or "").strip()
        if description:
            lines.append(f"- {tool.name}: {description}")
        else:
            lines.append(f"- {tool.name}")
    return "\n".join(lines)


def _build_messages(
    chat_history: Sequence[object], question: str, tools: Sequence[BaseTool]
) -> list[BaseMessage]:
    """Build message list: system + recent history + current question."""
    system = SYSTEM_PROMPT_MIXED if "Context from documents" in (question or "") else SYSTEM_PROMPT
    tool_summary = _build_tool_summary(tools)
    system = system.replace(
        TOOL_SUMMARY_PLACEHOLDER, tool_summary if tool_summary else "(No tools registered.)"
    )
    messages: list[BaseMessage] = [SystemMessage(content=system)]
    history = chat_history or []
    if MCP_MAX_HISTORY_MESSAGES > 0 and len(history) > MCP_MAX_HISTORY_MESSAGES:
        history = list(history[-MCP_MAX_HISTORY_MESSAGES:])
    for m in history:
        lc_msg = _message_to_langchain(m)
        if lc_msg is not None:
            messages.append(lc_msg)
    messages.append(HumanMessage(content=question))
    return messages


def _normalize_tool_args(tc: dict[str, object]) -> dict[str, object]:
    """Unwrap args like {'equation': {'value': '...'}} so tools receive plain values."""
    args_value = tc.get("args")
    if args_value in (None, ""):
        return {**tc, "args": {}}

    if isinstance(args_value, str):
        raw = args_value.strip()
        if not raw:
            return {**tc, "args": {}}
        try:
            parsed = cast(object, json.loads(raw))
        except (TypeError, ValueError):
            return {**tc, "args": {}}
        if not isinstance(parsed, Mapping):
            return {**tc, "args": {}}
        args_map = cast(Mapping[str, object], parsed)
    elif isinstance(args_value, Mapping):
        args_map = cast(Mapping[str, object], args_value)
    else:
        return {**tc, "args": {}}

    if set(args_map.keys()) == {"kwargs"}:
        kwargs_value = args_map.get("kwargs")
        if isinstance(kwargs_value, Mapping):
            args_map = cast(Mapping[str, object], kwargs_value)

    out: dict[str, object] = {}
    for key, value in args_map.items():
        key_str = key
        if isinstance(value, Mapping) and "value" in value:
            value_map = cast(Mapping[str, object], value)
            out[key_str] = value_map.get("value")
        else:
            out[key_str] = value
    return {**tc, "args": out}


def _tool_call_signature(name: str, args_map: Mapping[str, object] | None) -> str:
    payload = dict(args_map) if args_map is not None else {}
    return f"{name}:{json.dumps(payload, sort_keys=True, default=str)}"


def _run_coroutine_in_thread(coro: Coroutine[object, object, object]) -> object:
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("Thread runner did not return a value.")
    return result["value"]


async def _invoke_tool_async(tool: BaseTool, tool_call: dict[str, object]) -> object:
    tool_invoker = cast(_ToolInvoker, cast(object, tool))
    if hasattr(tool, "ainvoke"):
        return await tool_invoker.ainvoke(tool_call)
    return await asyncio.to_thread(tool_invoker.invoke, tool_call)


async def _invoke_llm_async(
    llm_with_tools: _ToolBound,
    messages: Sequence[BaseMessage],
    run_config: RunnableConfig | None,
) -> AIMessage:
    if hasattr(llm_with_tools, "ainvoke"):
        return await llm_with_tools.ainvoke(messages, config=run_config)
    return await asyncio.to_thread(llm_with_tools.invoke, messages, config=run_config)


def _tool_result_to_content(result: object) -> str:
    """MCP adapter may return ToolMessage; we need a string for ToolMessage.content."""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return getattr(result, "content", "") or ""
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


async def _get_mcp_answer_impl(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str]]:
    if get_mcp_settings().enable_mcp_tools is False:
        return "", []

    if tools is None:
        tools = await get_mcp_tools_async(server_keys=server_keys, run_config=run_config)
    if not tools:
        return "MCP tools are currently unavailable. Please try again.", []

    llm = cast(_ToolBindable, cast(object, get_llm(model_id=model_id)))
    tools_to_bind = list(tools[:1]) if require_tool_call and tools else list(tools)
    bind_kwargs: dict[str, object] = {}
    if require_tool_call and len(tools_to_bind) == 1:
        bind_kwargs["tool_choice"] = "required"
    llm_with_tools = llm.bind_tools(tools_to_bind, **bind_kwargs)
    tools_by_name = {t.name: t for t in tools_to_bind}
    messages = _build_messages(chat_history or [], question, tools_to_bind)
    if require_tool_call:
        available_tools = ", ".join(t.name for t in tools_to_bind)
        messages.insert(
            -1,
            SystemMessage(
                content=(
                    "Tool call required. You must call one available tool at least once before "
                    "providing a final answer. Do not answer directly."
                    f" Available tools: {available_tools}."
                )
            ),
        )
    tools_used: list[str] = []
    tool_call_attempts = 0
    tool_names = [t.name for t in tools_to_bind]
    last_tool_signature: str | None = None
    last_successful_tool_result: str | None = None
    logger.debug("MCP: available tools=%s for question='%s'", tool_names, question[:100])

    while True:
        try:
            ai = await _invoke_llm_async(llm_with_tools, messages, run_config)
        except ValueError as e:
            if "No generations found in stream" in str(e):
                logger.warning(
                    "MCP: OCI GenAI returned empty stream (tool call path); returning fallback: %s",
                    e,
                )
                return (
                    "The model returned an empty response. This can happen with tool calls; please try again.",
                    tools_used,
                )
            raise

        tool_calls = cast(list[dict[str, object]], getattr(ai, "tool_calls", None) or [])

        if not tool_calls:
            if require_tool_call and not tools_used and tool_call_attempts < 1:
                tool_call_attempts += 1
                messages.append(
                    SystemMessage(
                        content=(
                            "Tool call required. Respond with exactly one tool call using one of the "
                            f"available tools: {', '.join(tool_names)}."
                        )
                    )
                )
                continue

            content_raw = getattr(ai, "content", None)
            answer = (
                _normalize_message_content(cast(object, content_raw))
                if content_raw is not None
                else ""
            )
            if _looks_like_unresolved_tool_call(answer):
                if require_tool_call and not tools_used and tool_call_attempts < 1:
                    tool_call_attempts += 1
                    messages.append(
                        SystemMessage(
                            content=(
                                "Your previous response looked like a textual tool call instead of an actual "
                                "structured tool invocation. You must call exactly one available tool using the "
                                f"tool interface. Available tools: {', '.join(tool_names)}."
                            )
                        )
                    )
                    continue
                if not tools_used:
                    logger.warning("MCP: unresolved textual tool call rejected: %s", answer)
                    return (
                        "MCP tool call required but none was produced after retry. Please try again.",
                        tools_used,
                    )
            if require_tool_call and not tools_used:
                return (
                    "MCP tool call required but none was produced after retry. Please try again.",
                    tools_used,
                )
            logger.info(
                "mcp_answer_out answer_len=%s tools_used=%s",
                len(answer),
                tools_used,
            )
            return answer, tools_used

        if len(tool_calls) > 1:
            ai = AIMessage(
                content=_normalize_message_content(cast(object, ai.content)) or "",
                tool_calls=[tool_calls[0]],
                additional_kwargs=getattr(ai, "additional_kwargs", None) or {},
            )
            tool_calls = [tool_calls[0]]

        messages.append(ai)

        for tc in tool_calls:
            tc = _normalize_tool_args(tc)
            name_value = tc.get("name")
            name = name_value if isinstance(name_value, str) else "unknown"
            tool_call_id_value = tc.get("id")
            tool_call_id = (
                tool_call_id_value if isinstance(tool_call_id_value, str) else f"call_{name}"
            )
            tool = tools_by_name.get(name)

            if not tool:
                logger.warning("mcp_tool_unknown name=%s", name)
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {name}"}),
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )
                continue

            args_value = tc.get("args")
            args_map = (
                cast(Mapping[str, object], args_value) if isinstance(args_value, Mapping) else None
            )
            args_keys = list(args_map.keys()) if args_map is not None else []
            signature = _tool_call_signature(name, args_map)
            if signature == last_tool_signature:
                logger.warning(
                    "mcp_tool_repeat_detected name=%s tool_call_id=%s args_keys=%s",
                    name,
                    tool_call_id,
                    args_keys,
                )
                if last_successful_tool_result is not None:
                    logger.info(
                        "mcp_answer_out answer_len=%s tools_used=%s",
                        len(last_successful_tool_result),
                        tools_used,
                    )
                    return last_successful_tool_result, tools_used
                return (
                    "The model repeated the same tool call without progressing. Please try again.",
                    tools_used,
                )
            logger.info(
                "mcp_tool_call name=%s tool_call_id=%s args_keys=%s", name, tool_call_id, args_keys
            )
            try:
                result = await _invoke_tool_async(tool, tc)
                last_tool_signature = signature
                content = _tool_result_to_content(result)
                last_successful_tool_result = content
                if name not in tools_used:
                    tools_used.append(name)
                    llm_with_tools = llm.bind_tools(tools_to_bind)
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))
            except Exception as e:
                logger.warning("mcp_tool_error name=%s error=%s", name, e)
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )


def get_mcp_answer(
    question: str,
    chat_history: list[object] | None = None,
    model_id: str | None = None,
    server_keys: list[str] | None = None,
    tools: list[BaseTool] | None = None,
    require_tool_call: bool = False,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str]]:
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
        tuple[str, list[str]],
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
) -> tuple[str, list[str]]:
    return await _get_mcp_answer_impl(
        question,
        chat_history=chat_history,
        model_id=model_id,
        server_keys=server_keys,
        tools=tools,
        require_tool_call=require_tool_call,
        run_config=run_config,
    )

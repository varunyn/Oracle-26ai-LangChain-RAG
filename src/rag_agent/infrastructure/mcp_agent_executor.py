from __future__ import annotations

import inspect
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    LLMToolSelectorMiddleware,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from ..prompts.mcp_agent_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MIXED, TOOL_SUMMARY_PLACEHOLDER
from ..runtime.llm_invocation import suppress_llm_streaming
from .oci_models import get_llm

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MCPAnswerExecutionResult:
    answer: str
    tools_used: list[str]
    tool_invocations: list[dict[str, object]]
    state_messages: list[object]
_TOOL_SELECTOR_SYSTEM_PROMPT = """Select all tools that may be needed for the user's next step.

Use the tool names and descriptions exactly as provided.
Prefer a focused set, but include every plausibly relevant tool when the request is ambiguous.
For explicit workflows, include tools for queue discovery, per-item processing, supporting lookup/context, and finalization when those phases are relevant.
For data-grounded answers, keep oracle_retrieval available whenever it is present so the main model can ground the answer in the selected collection.
For math, database, CLI, or API requests, include the most specific tool plus any helper or inspection tool that may be needed to choose correct arguments.
For multi-part requests, select tools for every independent evidence or action need. A retrieval tool only covers collection facts; it does not cover computation, symbolic math, external actions, or API work.
Do not include tools that are clearly unrelated to the user's request."""


class OCIToolCallContentMiddleware(AgentMiddleware):
    """Keep LangChain tool calls out of OCI chat content blocks."""

    tools: Sequence[BaseTool] = ()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        return handler(_sanitize_model_request_for_oci(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        return await handler(_sanitize_model_request_for_oci(request))


def _sanitize_model_request_for_oci(request: ModelRequest) -> ModelRequest:
    messages = [_sanitize_ai_message_content_for_oci(message) for message in request.messages]
    if messages == request.messages:
        return request
    return request.override(messages=cast(Any, messages))


def _has_outer_callback_context(config: RunnableConfig | None) -> bool:
    return isinstance(config, dict) and config.get("callbacks") is not None


async def _run_graph_native_tool_loop(
    *,
    llm_model: Any,
    tools: Sequence[BaseTool],
    messages: Sequence[BaseMessage],
    system_prompt: str,
    config: RunnableConfig,
    max_rounds: int,
) -> Mapping[str, object]:
    bind_tools = getattr(llm_model, "bind_tools", None)
    if not callable(bind_tools):
        raise RuntimeError("Configured chat model does not support tool calling.")

    model_with_tools = bind_tools(list(tools))
    tool_node = ToolNode(list(tools))
    conversation: list[BaseMessage] = [SystemMessage(content=system_prompt), *messages]
    state_messages: list[object] = []
    rounds = max(1, max_rounds)

    for _idx in range(rounds):
        model_messages = [_sanitize_ai_message_content_for_oci(message) for message in conversation]
        response = await model_with_tools.ainvoke(cast(Any, model_messages), config=config)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=_normalize_message_content(getattr(response, "content", response)))
        conversation.append(response)
        state_messages.append(response)
        if not response.tool_calls:
            return {"messages": state_messages}

        tool_state = await tool_node.ainvoke({"messages": [response]}, config=config)
        raw_tool_messages = (
            tool_state.get("messages") if isinstance(tool_state, Mapping) else None
        )
        tool_messages = (
            list(cast(Sequence[object], raw_tool_messages))
            if isinstance(raw_tool_messages, Sequence)
            and not isinstance(raw_tool_messages, (str, bytes))
            else []
        )
        conversation.extend(cast(list[BaseMessage], tool_messages))
        state_messages.extend(tool_messages)

    raise ToolCallLimitExceededError("MCP agent stopped after tool call limit was reached.")


def _sanitize_agent_payload_for_oci(payload: dict[str, object]) -> dict[str, object]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return payload
    sanitized = [_sanitize_ai_message_content_for_oci(message) for message in messages]
    return {**payload, "messages": sanitized}


def _sanitize_ai_message_content_for_oci(message: object) -> object:
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message

    supported_content_types = {
        "text",
        "image_url",
        "document_url",
        "document",
        "file",
        "video_url",
        "video",
        "audio_url",
        "audio",
        "media",
    }
    content: list[str | dict[Any, Any]] = []
    for item in message.content:
        if isinstance(item, str):
            if item:
                content.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        content_type = item.get("type")
        if content_type == "tool_call":
            continue
        if content_type in supported_content_types:
            content.append(dict(item))
            continue
        if "text" in item and content_type is None:
            text = item.get("text")
            if isinstance(text, str) and text:
                content.append({"type": "text", "text": text})

    if not content and (message.tool_calls or message.additional_kwargs.get("tool_calls")):
        content = [{"type": "text", "text": "."}]
    if content == message.content:
        return message
    copy = getattr(message, "model_copy", None)
    if callable(copy):
        return copy(update={"content": content})
    return AIMessage(
        content=content,
        additional_kwargs=dict(message.additional_kwargs),
        response_metadata=dict(message.response_metadata),
        tool_calls=list(message.tool_calls),
        id=message.id,
        name=message.name,
    )


def _normalize_message_content(content: object) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text_value = item.get("text")
                parts.append(text_value if isinstance(text_value, str) else "")
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()


def _message_to_langchain(m: object) -> BaseMessage | None:
    if m is None:
        return None
    if isinstance(m, Mapping):
        role = str(m.get("role") or "").strip().lower()
        content = _normalize_message_content(m.get("content"))
        if not content:
            return None
        if role in {"assistant", "ai"}:
            return AIMessage(content=content)
        return HumanMessage(content=content)

    msg_type = str(getattr(m, "type", "") or getattr(m, "role", "")).strip().lower()
    content_attr = getattr(m, "content", None)
    if content_attr is None:
        return None
    content = _normalize_message_content(content_attr)
    if not content:
        return None
    if msg_type in {"assistant", "ai"}:
        return AIMessage(content=content)
    return HumanMessage(content=content)


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


def _is_mixed_mode(*, tools: Sequence[BaseTool], run_config: RunnableConfig | None) -> bool:
    configurable = (
        cast(dict[str, object], run_config.get("configurable"))
        if isinstance(run_config, dict) and isinstance(run_config.get("configurable"), dict)
        else {}
    )
    mode = str(configurable.get("mode") or "").strip().lower()
    if mode == "mixed":
        return True
    return any(getattr(tool, "name", "") == "oracle_retrieval" for tool in tools)


def _build_system_prompt(
    question: str, tools: Sequence[BaseTool], run_config: RunnableConfig | None
) -> str:
    base = (
        SYSTEM_PROMPT_MIXED if _is_mixed_mode(tools=tools, run_config=run_config) else SYSTEM_PROMPT
    )
    return base.replace(TOOL_SUMMARY_PLACEHOLDER, _build_tool_summary(tools))


def _build_messages(chat_history: Sequence[object] | None, question: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in chat_history or []:
        converted = _message_to_langchain(item)
        if converted is not None:
            messages.append(converted)
    messages.append(HumanMessage(content=question))
    return messages


def _message_signature(message: object) -> tuple[str, str] | None:
    if isinstance(message, HumanMessage):
        role = "human"
    elif isinstance(message, AIMessage):
        role = "ai"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        return None

    message_id = getattr(message, "id", None)
    if isinstance(message_id, str) and message_id.strip():
        return (role, f"id:{message_id.strip()}")
    content = _normalize_message_content(getattr(message, "content", ""))
    return (role, content)


def _strip_input_message_prefix(
    messages: Sequence[object],
    input_messages: Sequence[object],
) -> list[object]:
    remaining = list(messages)
    input_list = list(input_messages)
    if len(remaining) < len(input_list):
        return remaining
    for expected, actual in zip(input_list, remaining, strict=False):
        if _message_signature(expected) != _message_signature(actual):
            return remaining
    return remaining[len(input_list) :]


def _has_retrieval_tool(tools: Sequence[BaseTool]) -> bool:
    return any(str(getattr(tool, "name", "") or "").strip() == "oracle_retrieval" for tool in tools)


def _build_middleware(
    settings: object,
    tools: Sequence[BaseTool],
    *,
    tool_call_run_limit: int | None = None,
) -> list[object]:
    middleware: list[object] = []

    middleware.append(OCIToolCallContentMiddleware())
    middleware.append(ModelRetryMiddleware(max_retries=1))
    middleware.append(ToolRetryMiddleware(max_retries=1))
    if not _has_retrieval_tool(tools):
        middleware.append(
            LLMToolSelectorMiddleware(
                system_prompt=_TOOL_SELECTOR_SYSTEM_PROMPT,
            )
        )

    max_rounds = (
        tool_call_run_limit
        if tool_call_run_limit is not None
        else int(getattr(settings, "MCP_MAX_ROUNDS", 0) or 0)
    )
    if max_rounds > 0:
        middleware.append(ToolCallLimitMiddleware(run_limit=max_rounds, exit_behavior="error"))

    return middleware


def _extract_answer_and_tools(agent_state: Mapping[str, object]) -> tuple[str, list[str]]:
    messages_raw = agent_state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return "", []

    answer = ""
    tools_used: list[str] = []
    seen: set[str] = set()

    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, AIMessage):
            answer = _normalize_message_content(msg.content)
            raw_tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(raw_tool_calls, list):
                for tool_call in raw_tool_calls:
                    if not isinstance(tool_call, Mapping):
                        continue
                    tool_name = str(tool_call.get("name") or "").strip()
                    if tool_name and tool_name not in seen:
                        seen.add(tool_name)
                        tools_used.append(tool_name)
            continue
        if isinstance(msg, ToolMessage):
            tool_name = str(getattr(msg, "name", "") or "").strip()
            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                tools_used.append(tool_name)
            continue

    return answer, tools_used


_MAX_TOOL_TEXT = 24000
_MAX_JSON_DEPTH = 10
_MAX_JSON_KEYS = 80
_MAX_JSON_ITEMS = 200


def _truncate_tool_text(text: str, max_len: int = _MAX_TOOL_TEXT) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}\n… [{len(text)} characters total; truncated]"


def _jsonable_tool_value(value: object, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        return "<max depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= _MAX_JSON_KEYS:
                out["…"] = f"{len(value) - _MAX_JSON_KEYS} more keys"
                break
            out[str(k)] = _jsonable_tool_value(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)[:_MAX_JSON_ITEMS]
        return [_jsonable_tool_value(v, depth + 1) for v in items]
    return str(value)[:4000]


def _serialize_tool_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, ToolMessage):
        artifact = getattr(value, "artifact", None)
        if isinstance(artifact, Mapping):
            structured_content = artifact.get("structured_content")
            if structured_content is not None:
                return _serialize_tool_output(structured_content)
        return _truncate_tool_text(_normalize_message_content(value.content))
    if isinstance(value, str):
        return _truncate_tool_text(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            dumped = json.dumps(_jsonable_tool_value(value), ensure_ascii=True, default=str)
            return _truncate_tool_text(dumped)
        except Exception:  # noqa: BLE001
            return _truncate_tool_text(str(value))
    return _truncate_tool_text(str(value))


def _extract_tool_invocations(agent_state: Mapping[str, object]) -> list[dict[str, object]]:
    """Pair AIMessage tool_calls with ToolMessage results in conversation order."""
    messages_raw = agent_state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return []

    pending_by_id: dict[str, dict[str, object]] = {}
    orphan_ai_calls: deque[dict[str, object]] = deque()
    invocations: list[dict[str, object]] = []

    def _queue_ai_tool_call(*, name: str, tc_id: str, args: object) -> None:
        rec = {"tool_name": name, "args": args}
        if tc_id:
            pending_by_id[tc_id] = rec
        else:
            orphan_ai_calls.append(rec)

    def _with_tool_error_status(
        rec: dict[str, object], *, error_text: str | None
    ) -> dict[str, object]:
        if not error_text:
            return rec
        return {**rec, "error": error_text}

    def _complete_tool_result(
        *,
        tool_call_id: str,
        tool_name: str,
        content: object,
        status: str | None,
    ) -> None:
        text = _truncate_tool_text(_normalize_message_content(content))
        error_text = text if status == "error" else None
        if tool_call_id:
            rec = pending_by_id.pop(tool_call_id, None)
            if rec is None:
                invocations.append(
                    _with_tool_error_status(
                        {
                            "tool_name": tool_name,
                            "args": None,
                            "result": text,
                        },
                        error_text=error_text,
                    )
                )
            else:
                invocations.append(
                    _with_tool_error_status(
                        {**rec, "result": text},
                        error_text=error_text,
                    )
                )
            return
        if orphan_ai_calls:
            rec = orphan_ai_calls.popleft()
            invocations.append(
                _with_tool_error_status(
                    {**rec, "result": text},
                    error_text=error_text,
                )
            )
            return
        invocations.append(
            _with_tool_error_status(
                {
                    "tool_name": tool_name,
                    "args": None,
                    "result": text,
                },
                error_text=error_text,
            )
        )

    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, AIMessage):
            raw_calls = getattr(msg, "tool_calls", None)
            if isinstance(raw_calls, list):
                for tc in raw_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = str(tc.get("id") or "").strip()
                    name = str(tc.get("name") or "").strip()
                    if not name:
                        continue
                    raw_args = tc.get("args")
                    args = {} if raw_args is None else _jsonable_tool_value(raw_args)
                    _queue_ai_tool_call(name=name, tc_id=tc_id, args=args)
            continue

        if isinstance(msg, ToolMessage):
            tc_id = str(getattr(msg, "tool_call_id", "") or "").strip()
            content = getattr(msg, "content", "")
            name = str(getattr(msg, "name", "") or "").strip()
            status = str(getattr(msg, "status", "") or "").strip()
            _complete_tool_result(
                tool_call_id=tc_id,
                tool_name=name,
                content=content,
                status=status,
            )
            continue

    return invocations


async def get_mcp_answer_execution_with_langchain_agent_async(
    *,
    question: str,
    chat_history: Sequence[object] | None,
    model_id: str | None,
    tools: Sequence[BaseTool],
    run_config: RunnableConfig | None,
    require_tool_call: bool,
) -> MCPAnswerExecutionResult:
    if not tools:
        return MCPAnswerExecutionResult(
            answer="MCP tools are currently unavailable. Please try again.",
            tools_used=[],
            tool_invocations=[],
            state_messages=[],
        )

    # Local import keeps this module usable from tests without forcing full settings bootstrap.
    from api.settings import get_settings

    settings = get_settings()
    llm_model = get_llm(model_id=model_id)
    llm_model = suppress_llm_streaming(llm_model)

    try:
        current_messages: list[object] = cast(list[object], _build_messages(chat_history, question))
        response_state: Mapping[str, object] | None = None
        answer = ""
        tools_used: list[str] = []
        tool_invocations: list[dict[str, object]] = []
        state_messages: list[object] = []
        response_messages: list[object] = []

        invoke_config_map: dict[str, object] = dict(run_config or {})
        raw_callbacks = invoke_config_map.get("callbacks")
        if isinstance(raw_callbacks, list):
            invoke_config_map["callbacks"] = list(raw_callbacks)
        elif raw_callbacks is not None:
            invoke_config_map["callbacks"] = raw_callbacks
        invoke_config = cast(RunnableConfig, invoke_config_map)
        use_graph_native_tool_loop = _has_outer_callback_context(invoke_config)
        if not use_graph_native_tool_loop:
            raw_tags = invoke_config_map.get("tags")
            tags = list(raw_tags) if isinstance(raw_tags, list) else []
            if "nostream" not in tags:
                tags.append("nostream")
            invoke_config_map["tags"] = tags
            invoke_config = cast(RunnableConfig, invoke_config_map)
        agent = None
        if not use_graph_native_tool_loop:
            agent = create_agent(
                model=cast(Any, llm_model),
                tools=list(tools),
                system_prompt=_build_system_prompt(question, tools, run_config),
                middleware=cast(Any, _build_middleware(settings, tools)),
                name="mcp_agent_executor",
            )
        for retry_idx in range(2):
            try:
                if use_graph_native_tool_loop:
                    response_state = await _run_graph_native_tool_loop(
                        llm_model=llm_model,
                        tools=tools,
                        messages=cast(Sequence[BaseMessage], current_messages),
                        system_prompt=_build_system_prompt(question, tools, run_config),
                        config=invoke_config,
                        max_rounds=int(getattr(settings, "MCP_MAX_ROUNDS", 0) or 2),
                    )
                else:
                    response_state = cast(
                        Mapping[str, object],
                        await agent.ainvoke(
                            _sanitize_agent_payload_for_oci({"messages": current_messages}),
                            config=invoke_config,
                        ),
                    )
            except ToolCallLimitExceededError as exc:
                logger.info("MCP agent stopped after tool call limit was reached: %s", exc)
                return MCPAnswerExecutionResult(
                    answer=str(exc),
                    tools_used=[],
                    tool_invocations=[],
                    state_messages=state_messages,
                )
            answer, tools_used = _extract_answer_and_tools(response_state)
            tool_invocations = _extract_tool_invocations(response_state)
            raw_state_messages = response_state.get("messages")
            if isinstance(raw_state_messages, Sequence) and not isinstance(
                raw_state_messages, (str, bytes)
            ):
                response_messages = list(cast(Sequence[object], raw_state_messages))
                state_messages = _strip_input_message_prefix(
                    response_messages,
                    current_messages,
                )
            else:
                response_messages = []
                state_messages = []
            if retry_idx >= 1:
                break
            if require_tool_call and not tools_used:
                current_messages = (
                    list(response_messages)
                    if response_messages
                    else cast(list[object], _build_messages(chat_history, question))
                )
                current_messages.append(
                    HumanMessage(
                        "A real tool invocation is required for this request. "
                        "Call the best available tool based on its name, "
                        "description, and schema before giving the final answer."
                    )
                )
                continue
            break

        if require_tool_call and not tools_used:
            return MCPAnswerExecutionResult(
                answer="MCP tool call required but none was produced after retry. Please try again.",
                tools_used=[],
                tool_invocations=[],
                state_messages=state_messages,
            )
        return MCPAnswerExecutionResult(
            answer=answer,
            tools_used=tools_used,
            tool_invocations=tool_invocations,
            state_messages=state_messages,
        )
    finally:
        close = getattr(llm_model, "aclose", None)
        if callable(close):
            try:
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as exc:
                logger.debug("MCP agent LLM async cleanup failed: %s", exc)


async def get_mcp_answer_with_langchain_agent_async(
    *,
    question: str,
    chat_history: Sequence[object] | None,
    model_id: str | None,
    tools: Sequence[BaseTool],
    run_config: RunnableConfig | None,
    require_tool_call: bool,
) -> tuple[str, list[str], list[dict[str, object]]]:
    result = await get_mcp_answer_execution_with_langchain_agent_async(
        question=question,
        chat_history=chat_history,
        model_id=model_id,
        tools=tools,
        run_config=run_config,
        require_tool_call=require_tool_call,
    )
    return result.answer, result.tools_used, result.tool_invocations


__all__ = [
    "MCPAnswerExecutionResult",
    "get_mcp_answer_execution_with_langchain_agent_async",
    "get_mcp_answer_with_langchain_agent_async",
]

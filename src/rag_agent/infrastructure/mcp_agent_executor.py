from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ..prompts.mcp_agent_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MIXED, TOOL_SUMMARY_PLACEHOLDER
from .oci_models import get_llm

logger = logging.getLogger(__name__)

ToolProgressCallback = Callable[[dict[str, object]], None]
AnswerDeltaCallback = Callable[[str], None]
_TOOL_SELECTOR_SYSTEM_PROMPT = """Select all tools that may be needed for the user's next step.

Use the tool names and descriptions exactly as provided.
Prefer a focused set, but include every plausibly relevant tool when the request is ambiguous.
For explicit workflows, include tools for queue discovery, per-item processing, supporting lookup/context, and finalization when those phases are relevant.
For data-grounded answers, keep oracle_retrieval available whenever it is present so the main model can ground the answer in the selected collection.
For math, database, CLI, or API requests, include the most specific tool plus any helper or inspection tool that may be needed to choose correct arguments.
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
    return request.override(messages=messages)


async def _ainvoke_or_stream_agent(
    agent: Any,
    payload: dict[str, object],
    *,
    config: RunnableConfig,
    tool_progress_callback: ToolProgressCallback | None,
    answer_delta_callback: AnswerDeltaCallback | None = None,
    stop_after_tool_names: set[str] | None = None,
) -> Mapping[str, object]:
    payload = _sanitize_agent_payload_for_oci(payload)
    if (
        tool_progress_callback is None
        and answer_delta_callback is None
        and not stop_after_tool_names
        or not callable(getattr(agent, "astream_events", None))
    ):
        return cast(Mapping[str, object], await agent.ainvoke(cast(Any, payload), config=config))

    raw_stream = agent.astream_events(cast(Any, payload), config=config, version="v3")
    stream = await raw_stream if inspect.isawaitable(raw_stream) else raw_stream
    if not hasattr(stream, "tool_calls") or not hasattr(stream, "output"):
        raise RuntimeError(
            "LangChain stream_events(version='v3') did not expose typed stream projections."
        )

    if stop_after_tool_names:
        return await _consume_until_tool_result(
            stream,
            stop_after_tool_names=stop_after_tool_names,
            tool_progress_callback=tool_progress_callback,
            answer_delta_callback=answer_delta_callback,
        )

    consumers: list[Awaitable[None]] = []
    if tool_progress_callback is not None:
        consumers.append(_consume_tool_call_projection(stream, tool_progress_callback))
    if answer_delta_callback is not None and hasattr(stream, "__aiter__"):
        consumers.append(_consume_raw_text_events(stream, answer_delta_callback))
    if consumers:
        await asyncio.gather(*consumers)
    output = await _resolve_stream_output(stream)
    return cast(Mapping[str, object], output or {})


async def _consume_until_tool_result(
    stream: Any,
    *,
    stop_after_tool_names: set[str],
    tool_progress_callback: ToolProgressCallback | None,
    answer_delta_callback: AnswerDeltaCallback | None,
) -> Mapping[str, object]:
    latest_state: Mapping[str, object] = {}
    started_tool_ids: set[str] = set()
    ended_tool_ids: set[str] = set()
    normalized_stop_names = {name.strip().lower() for name in stop_after_tool_names if name.strip()}
    try:
        async for event in stream:
            if answer_delta_callback is not None:
                for text in _raw_text_deltas(event):
                    answer_delta_callback(text)
            state = _state_from_values_event(event)
            if state is None:
                continue
            latest_state = state
            if tool_progress_callback is not None:
                _emit_tool_progress_from_state(
                    state,
                    tool_progress_callback=tool_progress_callback,
                    started_tool_ids=started_tool_ids,
                    ended_tool_ids=ended_tool_ids,
                )
            for invocation in _extract_tool_invocations(state):
                tool_name = str(invocation.get("tool_name") or "").strip().lower()
                result = str(invocation.get("result") or invocation.get("error") or "").strip()
                if tool_name in normalized_stop_names and result:
                    return latest_state
    finally:
        close = getattr(stream, "aclose", None)
        if callable(close):
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result
    output = await _resolve_stream_output(stream)
    return cast(Mapping[str, object], output or latest_state or {})


async def _resolve_stream_output(stream: Any) -> object:
    output_attr = getattr(stream, "output", None)
    output = output_attr() if callable(output_attr) else output_attr
    return await output if inspect.isawaitable(output) else output


async def _drain_tool_call(call: Any) -> None:
    if hasattr(call, "__aiter__"):
        async for _delta in call:
            pass
        return

    output_deltas = getattr(call, "output_deltas", None)
    if hasattr(output_deltas, "__aiter__"):
        async for _delta in output_deltas:
            pass


async def _consume_tool_call_projection(
    stream: Any,
    tool_progress_callback: ToolProgressCallback,
) -> None:
    async for call in stream.tool_calls:
        tool_name = str(getattr(call, "tool_name", "") or "unknown_tool")
        tool_call_id = str(getattr(call, "tool_call_id", "") or "")
        tool_progress_callback(
            {
                "phase": "start",
                "tool_name": tool_name,
                "args": getattr(call, "input", None),
                "tool_run_id": tool_call_id,
            }
        )
        await _drain_tool_call(call)
        error = getattr(call, "error", None)
        if error:
            tool_progress_callback(
                {
                    "phase": "error",
                    "tool_name": tool_name,
                    "error": _truncate_tool_text(str(error)),
                    "tool_run_id": tool_call_id,
                }
            )
            continue
        tool_progress_callback(
            {
                "phase": "end",
                "tool_name": tool_name,
                "result": _serialize_tool_output(getattr(call, "output", None)),
                "tool_run_id": tool_call_id,
            }
        )


async def _consume_raw_text_events(
    stream: Any,
    answer_delta_callback: AnswerDeltaCallback,
) -> None:
    async for event in stream:
        for text in _raw_text_deltas(event):
            answer_delta_callback(text)


def _raw_text_deltas(event: object) -> list[str]:
    if not isinstance(event, Mapping) or event.get("method") != "messages":
        return []
    params = event.get("params")
    if not isinstance(params, Mapping):
        return []
    data = params.get("data")
    if not isinstance(data, tuple) or not data:
        return []
    chunk = data[0]
    if not isinstance(chunk, Mapping):
        return []
    if chunk.get("event") != "content-block-delta":
        return []
    delta = chunk.get("delta")
    if not isinstance(delta, Mapping) or delta.get("type") != "text-delta":
        return []
    text = str(delta.get("text") or "")
    return [text] if text else []


def _state_from_values_event(event: object) -> Mapping[str, object] | None:
    if not isinstance(event, Mapping) or event.get("method") != "values":
        return None
    params = event.get("params")
    if not isinstance(params, Mapping):
        return None
    data = params.get("data")
    return cast(Mapping[str, object], data) if isinstance(data, Mapping) else None


def _emit_tool_progress_from_state(
    state: Mapping[str, object],
    *,
    tool_progress_callback: ToolProgressCallback,
    started_tool_ids: set[str],
    ended_tool_ids: set[str],
) -> None:
    messages_raw = state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return
    tool_names_by_id: dict[str, str] = {}
    args_by_id: dict[str, object] = {}
    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, AIMessage):
            raw_calls = getattr(msg, "tool_calls", None)
            if not isinstance(raw_calls, list):
                continue
            for tool_call in raw_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_id = str(tool_call.get("id") or "").strip()
                tool_name = str(tool_call.get("name") or "").strip()
                if not tool_call_id or not tool_name:
                    continue
                tool_names_by_id[tool_call_id] = tool_name
                args_by_id[tool_call_id] = tool_call.get("args")
                if tool_call_id in started_tool_ids:
                    continue
                started_tool_ids.add(tool_call_id)
                tool_progress_callback(
                    {
                        "phase": "start",
                        "tool_name": tool_name,
                        "args": tool_call.get("args"),
                        "tool_run_id": tool_call_id,
                    }
                )
            continue
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id = str(getattr(msg, "tool_call_id", "") or "").strip()
        if not tool_call_id or tool_call_id in ended_tool_ids:
            continue
        ended_tool_ids.add(tool_call_id)
        status = str(getattr(msg, "status", "") or "").strip()
        tool_name = str(getattr(msg, "name", "") or "").strip() or tool_names_by_id.get(
            tool_call_id,
            "unknown_tool",
        )
        if status == "error":
            tool_progress_callback(
                {
                    "phase": "error",
                    "tool_name": tool_name,
                    "error": _truncate_tool_text(_serialize_tool_output(msg)),
                    "tool_run_id": tool_call_id,
                }
            )
            continue
        tool_progress_callback(
            {
                "phase": "end",
                "tool_name": tool_name,
                "result": _serialize_tool_output(msg),
                "tool_run_id": tool_call_id,
            }
        )


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
    content: list[object] = []
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


def _is_mixed_mode(
    *, tools: Sequence[BaseTool], run_config: RunnableConfig | None
) -> bool:
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
    base = SYSTEM_PROMPT_MIXED if _is_mixed_mode(tools=tools, run_config=run_config) else SYSTEM_PROMPT
    return base.replace(TOOL_SUMMARY_PLACEHOLDER, _build_tool_summary(tools))


def _build_messages(chat_history: Sequence[object] | None, question: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in chat_history or []:
        converted = _message_to_langchain(item)
        if converted is not None:
            messages.append(converted)
    messages.append(HumanMessage(content=question))
    return messages


def _tool_names_to_always_include(tools: Sequence[BaseTool]) -> list[str] | None:
    names = [str(getattr(tool, "name", "") or "").strip() for tool in tools]
    always_include = [name for name in names if name == "oracle_retrieval"]
    return always_include or None


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
    middleware.append(
        LLMToolSelectorMiddleware(
            system_prompt=_TOOL_SELECTOR_SYSTEM_PROMPT,
            always_include=_tool_names_to_always_include(tools),
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

    def _with_tool_error_status(rec: dict[str, object], *, error_text: str | None) -> dict[str, object]:
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


async def get_mcp_answer_with_langchain_agent_async(
    *,
    question: str,
    chat_history: Sequence[object] | None,
    model_id: str | None,
    tools: Sequence[BaseTool],
    run_config: RunnableConfig | None,
    require_tool_call: bool,
    tool_progress_callback: ToolProgressCallback | None = None,
    answer_delta_callback: AnswerDeltaCallback | None = None,
    stop_after_tool_names: set[str] | None = None,
) -> tuple[str, list[str], list[dict[str, object]]]:
    if not tools:
        return "MCP tools are currently unavailable. Please try again.", [], []

    # Local import keeps this module usable from tests without forcing full settings bootstrap.
    from api.settings import get_settings

    settings = get_settings()
    llm_model = get_llm(model_id=model_id)
    stream_answer_callback = None if require_tool_call else answer_delta_callback
    if stream_answer_callback is not None:
        try:
            setattr(llm_model, "is_stream", True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MCP agent LLM stream flag setup failed: %s", exc)

    try:
        agent = create_agent(
            model=cast(Any, llm_model),
            tools=list(tools),
            system_prompt=_build_system_prompt(question, tools, run_config),
            middleware=cast(Any, _build_middleware(settings, tools)),
            name="mcp_agent_executor",
        )
        current_messages: list[object] = cast(list[object], _build_messages(chat_history, question))
        response_state: Mapping[str, object] | None = None
        answer = ""
        tools_used: list[str] = []
        tool_invocations: list[dict[str, object]] = []

        invoke_config_map: dict[str, object] = dict(run_config or {})
        raw_callbacks = invoke_config_map.get("callbacks")
        callbacks = list(raw_callbacks) if isinstance(raw_callbacks, list) else []
        if callbacks:
            invoke_config_map["callbacks"] = callbacks
        invoke_config = cast(RunnableConfig, invoke_config_map)
        for retry_idx in range(2):
            try:
                response_state = cast(
                    Mapping[str, object],
                    await _ainvoke_or_stream_agent(
                        agent,
                        {"messages": current_messages},
                        config=invoke_config,
                        tool_progress_callback=tool_progress_callback,
                        answer_delta_callback=stream_answer_callback,
                        stop_after_tool_names=stop_after_tool_names,
                    ),
                )
            except ToolCallLimitExceededError as exc:
                logger.info("MCP agent stopped after tool call limit was reached: %s", exc)
                return str(exc), [], []
            answer, tools_used = _extract_answer_and_tools(response_state)
            tool_invocations = _extract_tool_invocations(response_state)
            if retry_idx >= 1:
                break
            if require_tool_call and not tools_used:
                state_messages = response_state.get("messages")
                current_messages = (
                    list(cast(Sequence[object], state_messages))
                    if isinstance(state_messages, Sequence)
                    and not isinstance(state_messages, (str, bytes))
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
            return "MCP tool call required but none was produced after retry. Please try again.", [], []
        return answer, tools_used, tool_invocations
    finally:
        close = getattr(llm_model, "aclose", None)
        if callable(close):
            try:
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as exc:
                logger.debug("MCP agent LLM async cleanup failed: %s", exc)


__all__ = ["get_mcp_answer_with_langchain_agent_async"]

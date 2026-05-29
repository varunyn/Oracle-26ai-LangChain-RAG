from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
import re
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from fractions import Fraction
from typing import Any, cast
from uuid import UUID

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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ..prompts.mcp_agent_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MIXED, TOOL_SUMMARY_PLACEHOLDER
from .oci_models import get_llm

logger = logging.getLogger(__name__)

_PSEUDO_TOOL_BLOCK = re.compile(
    r"<\|python_start\|>\s*([A-Za-z0-9_.]+)\((.*?)\)\s*<\|python_end\|>",
    re.DOTALL,
)
_CALC_EXPR_ARG = re.compile(r'expression\s*=\s*["\']([^"\']+)["\']')
_LITERAL_TOOL_CALL_RETRY_INSTRUCTION = (
    "You output literal tool-call text instead of an actual tool invocation. "
    "Continue from prior tool results and call tools directly (do not print "
    "tool_name(...)). Complete the task before giving the final answer."
)
ToolProgressCallback = Callable[[dict[str, object]], None]


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


def _tool_progress_from_stream_event(event: Mapping[str, object]) -> dict[str, object] | None:
    if event.get("method") != "tools":
        return None
    params = event.get("params")
    if not isinstance(params, Mapping):
        return None
    data = params.get("data")
    if not isinstance(data, Mapping):
        return None

    event_type = data.get("event")
    tool_call_id = str(data.get("tool_call_id") or "")
    tool_name = str(data.get("tool_name") or "").strip() or "unknown_tool"
    if event_type == "tool-started":
        return {
            "phase": "start",
            "tool_name": tool_name,
            "args": data.get("input"),
            "tool_run_id": tool_call_id,
        }
    if event_type == "tool-finished":
        return {
            "phase": "end",
            "tool_name": tool_name,
            "result": _serialize_tool_output(data.get("output")),
            "tool_run_id": tool_call_id,
        }
    if event_type == "tool-error":
        return {
            "phase": "error",
            "tool_name": tool_name,
            "error": _truncate_tool_text(str(data.get("message") or "Tool execution failed.")),
            "tool_run_id": tool_call_id,
        }
    return None


async def _ainvoke_or_stream_agent(
    agent: Any,
    payload: dict[str, object],
    *,
    config: RunnableConfig,
    tool_progress_callback: ToolProgressCallback | None,
) -> Mapping[str, object]:
    payload = _sanitize_agent_payload_for_oci(payload)
    if tool_progress_callback is None or not callable(getattr(agent, "astream_events", None)):
        return cast(Mapping[str, object], await agent.ainvoke(cast(Any, payload), config=config))

    raw_stream = agent.astream_events(cast(Any, payload), config=config, version="v3")
    stream = await raw_stream if inspect.isawaitable(raw_stream) else raw_stream
    if hasattr(stream, "tool_calls") and hasattr(stream, "output"):
        tool_task = asyncio.create_task(
            _consume_tool_call_projection(stream, tool_progress_callback)
        )
        try:
            output = await _resolve_stream_output(stream)
            await tool_task
        except BaseException:
            tool_task.cancel()
            with suppress(asyncio.CancelledError):
                await tool_task
            raise
        return cast(Mapping[str, object], output or {})

    latest_values: Mapping[str, object] | None = None
    async for event in stream:
        if not isinstance(event, Mapping):
            continue
        if event.get("method") == "values":
            params = event.get("params")
            if isinstance(params, Mapping) and isinstance(params.get("data"), Mapping):
                latest_values = cast(Mapping[str, object], params["data"])
            continue
        progress = _tool_progress_from_stream_event(event)
        if progress is not None:
            tool_progress_callback(progress)
    if latest_values is not None:
        return latest_values
    return cast(Mapping[str, object], await _resolve_stream_output(stream))


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


def _sanitize_agent_payload_for_oci(payload: dict[str, object]) -> dict[str, object]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return payload
    sanitized = [_sanitize_ai_message_content_for_oci(message) for message in messages]
    return {**payload, "messages": sanitized}


def _build_retry_messages_after_tool_error(
    *,
    chat_history: Sequence[object] | None,
    question: str,
    agent_state: Mapping[str, object],
) -> list[object]:
    """Summarize failed tool state without replaying provider-specific tool IDs."""
    messages = cast(list[object], _build_messages(chat_history, question))
    invocations = _extract_tool_invocations(agent_state)
    observations = (
        json.dumps(invocations, ensure_ascii=True)
        if invocations
        else "The prior tool attempt failed before producing a usable result."
    )
    messages.append(
        HumanMessage(
            "The prior tool attempt returned an error or unusable result. "
            f"Tool observations: {observations}\n"
            "Re-evaluate the request using the full available tool catalog. "
            "Choose the best tool based on its name, description, and schema, "
            "then answer from the tool result."
        )
    )
    return messages


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
    use_tool_selector: bool | None = None,
    use_tool_retry: bool = True,
    tool_call_run_limit: int | None = None,
) -> list[object]:
    middleware: list[object] = []
    selector_enabled = (
        bool(getattr(settings, "MCP_USE_LLM_TOOL_SELECTOR", False))
        if use_tool_selector is None
        else use_tool_selector
    )

    middleware.append(OCIToolCallContentMiddleware())
    middleware.append(ModelRetryMiddleware(max_retries=1))
    if use_tool_retry:
        middleware.append(ToolRetryMiddleware(max_retries=1))
    if selector_enabled:
        middleware.append(
            LLMToolSelectorMiddleware(always_include=_tool_names_to_always_include(tools))
        )

    max_rounds = (
        tool_call_run_limit
        if tool_call_run_limit is not None
        else int(getattr(settings, "MCP_MAX_ROUNDS", 0) or 0)
    )
    if max_rounds > 0:
        middleware.append(ToolCallLimitMiddleware(run_limit=max_rounds))

    return middleware


def _extract_answer_and_tools(agent_state: Mapping[str, object]) -> tuple[str, list[str]]:
    messages_raw = agent_state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return "", []

    answer = ""
    tools_used: list[str] = []
    seen: set[str] = set()

    def _collect_tool_names(raw_tool_calls: object) -> list[str]:
        if not isinstance(raw_tool_calls, list):
            return []
        names: list[str] = []
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name:
                function = tool_call.get("function")
                if isinstance(function, Mapping):
                    tool_name = str(function.get("name") or "").strip()
            if tool_name:
                names.append(tool_name)
        return names

    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, AIMessage):
            answer = _normalize_message_content(msg.content)
            additional_kwargs = getattr(msg, "additional_kwargs", None)
            response_metadata = getattr(msg, "response_metadata", None)
            ai_candidates: list[object] = [
                getattr(msg, "tool_calls", None),
                (
                    additional_kwargs.get("tool_calls")
                    if isinstance(additional_kwargs, Mapping)
                    else None
                ),
                (
                    response_metadata.get("tool_calls")
                    if isinstance(response_metadata, Mapping)
                    else None
                ),
            ]
            for candidate in ai_candidates:
                for tool_name in _collect_tool_names(candidate):
                    if tool_name not in seen:
                        seen.add(tool_name)
                        tools_used.append(tool_name)
        if isinstance(msg, ToolMessage):
            tool_name = str(getattr(msg, "name", "") or "").strip()
            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                tools_used.append(tool_name)
            continue
        if isinstance(msg, Mapping):
            msg_type = str(msg.get("type") or msg.get("role") or "").strip().lower()
            if msg_type in {"ai", "assistant"}:
                content = msg.get("content")
                answer = _normalize_message_content(content)
                dict_candidates: list[object] = [
                    msg.get("tool_calls"),
                    (
                        msg.get("additional_kwargs", {}).get("tool_calls")
                        if isinstance(msg.get("additional_kwargs"), Mapping)
                        else None
                    ),
                    (
                        msg.get("response_metadata", {}).get("tool_calls")
                        if isinstance(msg.get("response_metadata"), Mapping)
                        else None
                    ),
                ]
                for candidate in dict_candidates:
                    for tool_name in _collect_tool_names(candidate):
                        if tool_name not in seen:
                            seen.add(tool_name)
                            tools_used.append(tool_name)
            elif msg_type == "tool":
                tool_name = str(msg.get("name") or "").strip()
                if tool_name and tool_name not in seen:
                    seen.add(tool_name)
                    tools_used.append(tool_name)

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


def _normalize_tool_args(raw: object) -> object:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return _jsonable_tool_value(parsed)
            except Exception:  # noqa: BLE001
                return raw
        return raw
    return _jsonable_tool_value(raw)


def _parse_one_tool_call(tc: Mapping[str, object]) -> tuple[str, str, object]:
    tc_id = str(tc.get("id") or "").strip()
    name = str(tc.get("name") or "").strip()
    args: object = tc.get("args")
    if args is None and "arguments" in tc:
        args = tc.get("arguments")
    fn = tc.get("function")
    if isinstance(fn, Mapping):
        if not name:
            name = str(fn.get("name") or "").strip()
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:  # noqa: BLE001
                args = raw_args
        elif raw_args is not None:
            args = raw_args
    if args is None:
        args = {}
    return name, tc_id, _normalize_tool_args(args)


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


def _normalize_tool_start_args(input_str: object, inputs: object) -> object:
    if isinstance(inputs, Mapping):
        return _normalize_tool_args(dict(inputs))
    if isinstance(input_str, str) and input_str.strip():
        stripped = input_str.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return _normalize_tool_args(stripped)
        return stripped
    return None


class _ToolProgressCallback(BaseCallbackHandler):
    _on_event: ToolProgressCallback
    _run_context: dict[str, dict[str, object]]

    def __init__(self, on_event: ToolProgressCallback) -> None:
        super().__init__()
        self._on_event = on_event
        self._run_context = {}

    def _emit(self, payload: dict[str, object]) -> None:
        try:
            self._on_event(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("tool progress callback failed: %s", exc)

    def on_tool_start(
        self,
        serialized: dict[str, object],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        _ = parent_run_id, kwargs
        tool_name = str(serialized.get("name") or serialized.get("id") or "").strip()
        if not tool_name:
            tool_name = "unknown_tool"
        args = _normalize_tool_start_args(input_str, inputs)
        run_key = str(run_id)
        self._run_context[run_key] = {
            "tool_name": tool_name,
            "args": args,
        }
        self._emit(
            {
                "phase": "start",
                "tool_name": tool_name,
                "args": args,
                "tool_run_id": run_key,
            }
        )

    def on_tool_end(
        self,
        output: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        _ = parent_run_id, kwargs
        run_key = str(run_id)
        context = self._run_context.pop(run_key, {})
        self._emit(
            {
                "phase": "end",
                "tool_name": str(context.get("tool_name") or "unknown_tool"),
                "args": context.get("args"),
                "result": _serialize_tool_output(output),
                "tool_run_id": run_key,
            }
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        _ = parent_run_id, kwargs
        run_key = str(run_id)
        context = self._run_context.pop(run_key, {})
        self._emit(
            {
                "phase": "error",
                "tool_name": str(context.get("tool_name") or "unknown_tool"),
                "args": context.get("args"),
                "error": _truncate_tool_text(str(error)),
                "tool_run_id": run_key,
            }
        )


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

    def _complete_tool_result(*, tool_call_id: str, tool_name: str, content: object) -> None:
        text = _truncate_tool_text(_normalize_message_content(content))
        if tool_call_id:
            rec = pending_by_id.pop(tool_call_id, None)
            if rec is None:
                invocations.append(
                    {
                        "tool_name": tool_name,
                        "args": None,
                        "result": text,
                    }
                )
            else:
                invocations.append({**rec, "result": text})
            return
        if orphan_ai_calls:
            rec = orphan_ai_calls.popleft()
            invocations.append({**rec, "result": text})
            return
        invocations.append(
            {
                "tool_name": tool_name,
                "args": None,
                "result": text,
            }
        )

    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, AIMessage):
            raw_calls = getattr(msg, "tool_calls", None)
            if isinstance(raw_calls, list):
                for tc in raw_calls:
                    if not isinstance(tc, dict):
                        continue
                    name, tc_id, args = _parse_one_tool_call(tc)
                    if not name:
                        continue
                    _queue_ai_tool_call(name=name, tc_id=tc_id, args=args)
            continue

        if isinstance(msg, ToolMessage):
            tc_id = str(getattr(msg, "tool_call_id", "") or "").strip()
            content = getattr(msg, "content", "")
            name = str(getattr(msg, "name", "") or "").strip()
            _complete_tool_result(tool_call_id=tc_id, tool_name=name, content=content)
            continue

        if isinstance(msg, Mapping):
            msg_type = str(msg.get("type") or msg.get("role") or "").strip().lower()
            if msg_type in {"ai", "assistant"}:
                raw_calls = msg.get("tool_calls")
                if isinstance(raw_calls, list):
                    for tc in raw_calls:
                        if not isinstance(tc, dict):
                            continue
                        name, tc_id, args = _parse_one_tool_call(tc)
                        if not name:
                            continue
                        _queue_ai_tool_call(name=name, tc_id=tc_id, args=args)
            elif msg_type == "tool":
                tc_id = str(msg.get("tool_call_id") or msg.get("toolCallId") or "").strip()
                content = msg.get("content", "")
                name = str(msg.get("name") or "").strip()
                _complete_tool_result(tool_call_id=tc_id, tool_name=name, content=content)

    return invocations


def _safe_eval_arithmetic(expr: str) -> Fraction:
    node = ast.parse(expr, mode="eval")

    def _visit(n: ast.AST) -> Fraction:
        if isinstance(n, ast.Expression):
            return _visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if isinstance(n.value, int):
                return Fraction(n.value, 1)
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _visit(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _visit(n.left)
            right = _visit(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Unsupported expression")

    return _visit(node)


def _clean_leaked_tool_syntax(answer: str, tools_used: Sequence[str]) -> str:
    if tools_used:
        return answer
    raw = answer.strip()
    if not raw:
        return raw
    match = _PSEUDO_TOOL_BLOCK.search(raw)
    if not match:
        return raw

    tool_name = match.group(1).strip()
    args_block = match.group(2)
    if "calculator" in tool_name.lower():
        expr_match = _CALC_EXPR_ARG.search(args_block)
        if expr_match:
            expr = expr_match.group(1).strip()
            try:
                value = _safe_eval_arithmetic(expr)
                if value.denominator == 1:
                    return str(value.numerator)
                return f"{value.numerator}/{value.denominator}"
            except Exception:  # noqa: BLE001
                pass

    cleaned = _PSEUDO_TOOL_BLOCK.sub("", raw).strip()
    return cleaned or raw


def _extract_literal_tool_call_names(answer: str, tool_names: Sequence[str]) -> set[str]:
    if not answer.strip() or not tool_names:
        return set()
    found: set[str] = set()
    for tool_name in tool_names:
        escaped = re.escape(tool_name)
        if re.search(rf"\b{escaped}\s*\(", answer):
            found.add(tool_name)
    return found


def _should_retry_for_literal_tool_text(
    *,
    answer: str,
    tools_used: Sequence[str],
    tools: Sequence[BaseTool],
) -> bool:
    if not answer.strip():
        return False
    tool_names = [str(getattr(tool, "name", "") or "").strip() for tool in tools]
    tool_names = [name for name in tool_names if name]
    if not tool_names:
        return False

    mentioned_tools = _extract_literal_tool_call_names(answer, tool_names)
    if not mentioned_tools:
        return False

    used = {name.strip() for name in tools_used if name.strip()}
    return any(name not in used for name in mentioned_tools)


def _tool_message_has_error(msg: object) -> bool:
    status = str(getattr(msg, "status", "") or "").strip().lower()
    if status == "error":
        return True

    artifact = getattr(msg, "artifact", None)
    if isinstance(artifact, Mapping):
        structured_content = artifact.get("structured_content")
        if isinstance(structured_content, Mapping) and "error" in structured_content:
            return True
        if "error" in artifact:
            return True

    content = _normalize_message_content(getattr(msg, "content", ""))
    stripped = content.strip()
    if not stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:  # noqa: BLE001
        return "tool call limit exceeded" in stripped.lower()
    return isinstance(parsed, Mapping) and "error" in parsed


def _agent_state_has_tool_error(agent_state: Mapping[str, object]) -> bool:
    messages_raw = agent_state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return False

    for msg in cast(Sequence[object], messages_raw):
        if isinstance(msg, ToolMessage) and _tool_message_has_error(msg):
            return True
        if isinstance(msg, Mapping):
            msg_type = str(msg.get("type") or msg.get("role") or "").strip().lower()
            if msg_type != "tool":
                continue
            status = str(msg.get("status") or "").strip().lower()
            if status == "error":
                return True
            content = _normalize_message_content(msg.get("content", ""))
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                if "tool call limit exceeded" in content.lower():
                    return True
                continue
            if isinstance(parsed, Mapping) and "error" in parsed:
                return True
    return False


def _normalize_ai_tool_call_ids(agent_state: Mapping[str, object]) -> None:
    messages_raw = agent_state.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(messages_raw, (str, bytes)):
        return

    for message in cast(Sequence[object], messages_raw):
        if not isinstance(message, AIMessage):
            continue
        raw_tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(raw_tool_calls, list):
            continue

        normalized_ids: list[str] = []
        for idx, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, dict):
                continue
            current_id = tool_call.get("id")
            if isinstance(current_id, str) and current_id.strip():
                normalized_ids.append(current_id.strip())
                continue
            generated_id = f"call_{idx}_{uuid.uuid4().hex[:12]}"
            tool_call["id"] = generated_id
            normalized_ids.append(generated_id)

        if not normalized_ids:
            continue

        for container_name in ("additional_kwargs", "response_metadata"):
            container = getattr(message, container_name, None)
            if not isinstance(container, dict):
                continue
            container_tool_calls = container.get("tool_calls")
            if not isinstance(container_tool_calls, list):
                continue
            for idx, tool_call in enumerate(container_tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                existing = tool_call.get("id")
                if isinstance(existing, str) and existing.strip():
                    continue
                if idx < len(normalized_ids):
                    tool_call["id"] = normalized_ids[idx]


async def get_mcp_answer_with_langchain_agent_async(
    *,
    question: str,
    chat_history: Sequence[object] | None,
    model_id: str | None,
    tools: Sequence[BaseTool],
    run_config: RunnableConfig | None,
    require_tool_call: bool,
    tool_progress_callback: ToolProgressCallback | None = None,
) -> tuple[str, list[str], list[dict[str, object]]]:
    if not tools:
        return "MCP tools are currently unavailable. Please try again.", [], []

    # Local import keeps this module usable from tests without forcing full settings bootstrap.
    from api.settings import get_settings

    settings = get_settings()
    llm_model = get_llm(model_id=model_id)

    def _create_mcp_agent(
        *,
        use_tool_selector: bool | None = None,
        use_tool_retry: bool = True,
        tool_call_run_limit: int | None = None,
    ) -> Any:
        return create_agent(
            model=cast(Any, llm_model),
            tools=list(tools),
            system_prompt=_build_system_prompt(question, tools, run_config),
            middleware=cast(
                Any,
                _build_middleware(
                    settings,
                    tools,
                    use_tool_selector=use_tool_selector,
                    use_tool_retry=use_tool_retry,
                    tool_call_run_limit=tool_call_run_limit,
                ),
            ),
            name="mcp_agent_executor",
        )

    try:
        agent = _create_mcp_agent()
        current_messages: list[object] = cast(list[object], _build_messages(chat_history, question))
        response_state: Mapping[str, object] | None = None
        answer = ""
        tools_used: list[str] = []
        tool_invocations: list[dict[str, object]] = []
        retried_after_tool_error = False

        invoke_config_map: dict[str, object] = dict(run_config or {})
        raw_callbacks = invoke_config_map.get("callbacks")
        callbacks = list(raw_callbacks) if isinstance(raw_callbacks, list) else []
        agent_supports_stream_events = callable(getattr(agent, "astream_events", None))
        if tool_progress_callback is not None and not agent_supports_stream_events:
            callbacks.append(_ToolProgressCallback(tool_progress_callback))
        if callbacks:
            invoke_config_map["callbacks"] = callbacks
        invoke_config = cast(RunnableConfig, invoke_config_map)
        for retry_idx in range(2):
            response_state = cast(
                Mapping[str, object],
                await _ainvoke_or_stream_agent(
                    agent,
                    {"messages": current_messages},
                    config=invoke_config,
                    tool_progress_callback=tool_progress_callback,
                ),
            )
            _normalize_ai_tool_call_ids(response_state)
            answer, tools_used = _extract_answer_and_tools(response_state)
            tool_invocations = _extract_tool_invocations(response_state)
            answer = _clean_leaked_tool_syntax(answer, tools_used)
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
            if (
                tools_used
                and not retried_after_tool_error
                and _agent_state_has_tool_error(response_state)
            ):
                retried_after_tool_error = True
                current_messages = _build_retry_messages_after_tool_error(
                    chat_history=chat_history,
                    question=question,
                    agent_state=response_state,
                )
                continue
            if not _should_retry_for_literal_tool_text(
                answer=answer,
                tools_used=tools_used,
                tools=tools,
            ):
                break

            state_messages = response_state.get("messages")
            if not isinstance(state_messages, Sequence) or isinstance(state_messages, (str, bytes)):
                break
            current_messages = [
                *cast(list[object], list(state_messages)),
                HumanMessage(_LITERAL_TOOL_CALL_RETRY_INSTRUCTION),
            ]

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

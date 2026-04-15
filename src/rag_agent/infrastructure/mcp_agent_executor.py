from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
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


def _build_middleware(settings: object, llm_model: object) -> list[object]:
    middleware: list[object] = []

    max_tools = int(getattr(settings, "MCP_TOOL_SELECTION_MAX_TOOLS", 0) or 0)
    always_raw = getattr(settings, "MCP_TOOL_SELECTION_ALWAYS_INCLUDE", None) or []
    always_include = [str(v).strip() for v in always_raw if str(v).strip()]
    llm_module = str(getattr(type(llm_model), "__module__", "") or "")
    # OCI chat models currently fail inside LLMToolSelector structured-output path.
    # Keep runtime stable by skipping selector middleware for OCI providers.
    supports_llm_selector = not llm_module.startswith("langchain_oci.")
    if max_tools > 0 and supports_llm_selector:
        middleware.append(
            LLMToolSelectorMiddleware(
                model=cast(Any, llm_model),
                max_tools=max_tools,
                always_include=always_include or None,
            )
        )
    elif max_tools > 0 and not supports_llm_selector:
        logger.info(
            "MCP: skipping LLMToolSelectorMiddleware for model module %s (provider limitation)",
            llm_module,
        )

    middleware.append(ModelRetryMiddleware(max_retries=1))
    middleware.append(ToolRetryMiddleware(max_retries=1))

    max_rounds = int(getattr(settings, "MCP_MAX_ROUNDS", 0) or 0)
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
            ai_candidates: list[object] = [
                getattr(msg, "tool_calls", None),
                (
                    getattr(msg, "additional_kwargs", None).get("tool_calls")
                    if isinstance(getattr(msg, "additional_kwargs", None), Mapping)
                    else None
                ),
                (
                    getattr(msg, "response_metadata", None).get("tool_calls")
                    if isinstance(getattr(msg, "response_metadata", None), Mapping)
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
) -> tuple[str, list[str], list[dict[str, object]]]:
    if not tools:
        return "MCP tools are currently unavailable. Please try again.", [], []

    # Local import keeps this module usable from tests without forcing full settings bootstrap.
    from api.settings import get_settings

    settings = get_settings()
    llm_model = get_llm(model_id=model_id)
    agent = create_agent(
        model=cast(Any, llm_model),
        tools=list(tools),
        system_prompt=_build_system_prompt(question, tools, run_config),
        middleware=cast(Any, _build_middleware(settings, llm_model)),
        name="mcp_agent_executor",
    )
    response_state = cast(
        Mapping[str, object],
        await agent.ainvoke(
            {"messages": _build_messages(chat_history, question)},
            config=run_config,
        ),
    )
    _normalize_ai_tool_call_ids(response_state)
    answer, tools_used = _extract_answer_and_tools(response_state)
    tool_invocations = _extract_tool_invocations(response_state)
    answer = _clean_leaked_tool_syntax(answer, tools_used)
    if require_tool_call and not tools_used:
        return "MCP tool call required but none was produced after retry. Please try again.", [], []
    return answer, tools_used, tool_invocations


__all__ = ["get_mcp_answer_with_langchain_agent_async"]

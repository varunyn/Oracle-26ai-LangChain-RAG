import asyncio
import logging
import re
import time
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from ...agent_state import State, ensure_required
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.direct_mcp_tools import get_mcp_tools, get_mcp_tools_async
from ...infrastructure.oci_models import get_llm
from ...utils.context_window import (
    calculate_context_usage,
    log_context_usage,
    messages_to_text,
)

logger = logging.getLogger(__name__)

_RAW_TOOL_CALL_PATTERN = re.compile(r"^[\w\.]+\s*\([^)]*\)$")


def _looks_like_unresolved_tool_call(text: str) -> bool:
    candidate = text.strip()
    return bool(candidate) and len(candidate) <= 200 and bool(_RAW_TOOL_CALL_PATTERN.fullmatch(candidate))


class SelectMCPTools(Runnable[State, dict[str, object]]):
    """
    Ensures MCP path is set up; server_keys come from config at invoke time.
    Increments round for MoreInfo? loop cap.

    Best practices:
    - Simple state preparation node
    - Supports both sync and async invocation
    """

    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Sync invocation."""
        return self._select(input, config)

    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Async invocation (same logic for state updates)."""
        return self._select(input, config)

    def _select(self, input: State, config: RunnableConfig | None = None) -> dict[str, object]:
        """Internal selection logic."""
        log_node_start("SelectMCPTools")
        round_val = input.get("round") or 0
        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        max_rounds = run_config.get("configurable", {}).get("max_rounds", 2)
        logger.debug("SelectMCPTools: round=%d, max_rounds=%d", round_val + 1, max_rounds)
        log_node_end("SelectMCPTools", round=round_val + 1, max_rounds=max_rounds)
        return {
            "round": round_val + 1,
            "max_rounds": max_rounds,
            "selected_mcp_tool_names": list(input.get("selected_mcp_tool_names") or []),
            "selected_mcp_tool_descriptions": list(
                input.get("selected_mcp_tool_descriptions") or []
            ),
        }


def _ensure_message_list(messages_in: Sequence[object]) -> list[HumanMessage | AIMessage]:
    """Normalize messages (dicts or BaseMessage) to HumanMessage/AIMessage list."""
    out: list[HumanMessage | AIMessage] = []
    for m in messages_in or []:
        if m is None:
            continue
        if isinstance(m, dict):
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role in ("human", "user"):
                out.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                out.append(AIMessage(content=content))
            elif content:
                out.append(HumanMessage(content=content))
        elif hasattr(m, "content"):
            # Assume already a BaseMessage-compatible object
            out.append(cast(HumanMessage | AIMessage, m))
    return out


def _compute_context_usage(
    question: str,
    messages: Sequence[object],
    model_id: str | None,
    history_text: str | None = None,
) -> dict[str, Any]:
    cached = (history_text or "").strip()
    if cached:
        context_text = f"{cached}\nhuman: {question}" if question else cached
    else:
        msg_list = _ensure_message_list(messages)
        if question:
            msg_list.append(HumanMessage(content=question))
        context_text = messages_to_text(msg_list)
    usage = calculate_context_usage(context_text, model_id)
    log_context_usage(usage)
    return usage


class CallMCPTools(Runnable[State, dict[str, object]]):
    @staticmethod
    def _selected_tool_names(input: State) -> list[str]:
        raw_tools = input.get("selected_mcp_tool_names")
        if not isinstance(raw_tools, list):
            return []
        selected: list[str] = []
        for raw_name in raw_tools:
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if name:
                selected.append(name)
        return selected

    @staticmethod
    def _filter_selected_tools(
        tools: Sequence[BaseTool],
        selected_tool_names: Sequence[str],
    ) -> list[BaseTool]:
        selected_lookup = set(selected_tool_names)
        return [tool for tool in tools if tool.name in selected_lookup]

    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Sync MCP tool invocation. Only runs on MCP path (RAG and tools never run in same turn)."""
        log_node_start("CallMCPTools")
        ensure_required(input, "CallMCPTools")
        t0 = time.perf_counter()
        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        model_id = configurable.get("model_id")
        server_keys = cast(list[str] | None, configurable.get("mcp_server_keys"))
        mode = (input.get("mode") or "").lower()

        question = input.get("user_request", "")
        history_messages = input.get("messages", [])
        history_text = cast(str | None, input.get("history_text"))
        selected_tool_names = self._selected_tool_names(input)
        tools_for_agent: list[BaseTool] | None = None

        if selected_tool_names:
            loaded_tools = get_mcp_tools(server_keys=server_keys, run_config=run_config)
            tools_for_agent = self._filter_selected_tools(loaded_tools, selected_tool_names)

        require_tool_call = mode == "mcp" or (mode == "mixed" and bool(selected_tool_names))

        logger.info(
            "CallMCPTools: mode=%s selected_tools=%s resolved_tools=%d require_tool_call=%s",
            mode,
            selected_tool_names,
            len(tools_for_agent or []),
            require_tool_call,
        )

        try:
            answer, tools_used = self._invoke_get_mcp_answer(
                question,
                history_messages,
                model_id,
                server_keys,
                tools_for_agent,
                require_tool_call=require_tool_call,
                run_config=run_config,
            )
        except Exception as e:
            logger.error("CallMCPTools error: %s", e, exc_info=True)
            log_node_end(
                "CallMCPTools", duration_ms=(time.perf_counter() - t0) * 1000, error=str(e)
            )
            return {
                "mcp_answer": "",
                "mcp_tools_used": [],
                "mcp_used": False,
                "error": f"MCP tool invocation failed: {str(e)}",
                "context_usage": _compute_context_usage(
                    question, history_messages, model_id, history_text
                ),
            }

        duration_ms = (time.perf_counter() - t0) * 1000
        log_node_end(
            "CallMCPTools", duration_ms=duration_ms, tools_used_count=len(tools_used or [])
        )
        if not tools_used and _looks_like_unresolved_tool_call(answer or ""):
            logger.warning("CallMCPTools: unresolved textual tool call rejected: %s", answer)
            return {
                "mcp_answer": "",
                "mcp_tools_used": [],
                "mcp_used": False,
                "latest_answer": "",
                "error": "Unresolved MCP tool call. Please try again.",
                "context_usage": _compute_context_usage(
                    question, history_messages, model_id, history_text
                ),
            }
        return {
            "mcp_answer": answer or "",
            "mcp_tools_used": tools_used or [],
            "mcp_used": bool(tools_used),
            "latest_answer": answer or "",
            "context_usage": _compute_context_usage(
                question, history_messages, model_id, history_text
            ),
        }

    def _invoke_get_mcp_answer(
        self,
        question: str,
        messages: Sequence[object],
        model_id: str | None,
        server_keys: list[str] | None,
        tools: list[BaseTool] | None = None,
        require_tool_call: bool = False,
        run_config: RunnableConfig | None = None,
    ) -> tuple[str, list[str]]:
        """Invoke MCP via get_mcp_answer (no RAG context; this path is tools-only)."""
        try:
            from ...infrastructure.mcp_agent import get_mcp_answer
        except ImportError:
            logger.warning("MCP: get_mcp_answer not available")
            return "", []
        return get_mcp_answer(
            question,
            list(messages),
            model_id=model_id,
            server_keys=server_keys,
            tools=tools,
            require_tool_call=require_tool_call,
            run_config=run_config,
        )

    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        log_node_start("CallMCPTools")
        ensure_required(input, "CallMCPTools")
        t0 = time.perf_counter()
        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        model_id = configurable.get("model_id")
        server_keys = cast(list[str] | None, configurable.get("mcp_server_keys"))
        mode = (input.get("mode") or "").lower()

        question = input.get("user_request", "")
        history_messages = input.get("messages", [])
        history_text = cast(str | None, input.get("history_text"))
        selected_tool_names = self._selected_tool_names(input)
        tools_for_agent: list[BaseTool] | None = None

        if selected_tool_names:
            loaded_tools = await get_mcp_tools_async(server_keys=server_keys, run_config=run_config)
            tools_for_agent = self._filter_selected_tools(loaded_tools, selected_tool_names)

        require_tool_call = mode == "mcp" or (mode == "mixed" and bool(selected_tool_names))

        logger.info(
            "CallMCPTools: mode=%s selected_tools=%s resolved_tools=%d async require_tool_call=%s",
            mode,
            selected_tool_names,
            len(tools_for_agent or []),
            require_tool_call,
        )
        try:
            from ...infrastructure.mcp_agent import get_mcp_answer_async

            answer, tools_used = await get_mcp_answer_async(
                question,
                list(history_messages),
                model_id=model_id,
                server_keys=server_keys,
                tools=tools_for_agent,
                require_tool_call=require_tool_call,
                run_config=run_config,
            )
        except Exception as e:
            logger.error("CallMCPTools error: %s", e, exc_info=True)
            log_node_end(
                "CallMCPTools", duration_ms=(time.perf_counter() - t0) * 1000, error=str(e)
            )
            return {
                "mcp_answer": "",
                "mcp_tools_used": [],
                "mcp_used": False,
                "error": f"MCP tool invocation failed: {str(e)}",
                "context_usage": _compute_context_usage(
                    question, history_messages, model_id, history_text
                ),
            }

        duration_ms = (time.perf_counter() - t0) * 1000
        log_node_end(
            "CallMCPTools", duration_ms=duration_ms, tools_used_count=len(tools_used or [])
        )
        return {
            "mcp_answer": answer or "",
            "mcp_tools_used": tools_used or [],
            "mcp_used": bool(tools_used),
            "latest_answer": answer or "",
            "context_usage": _compute_context_usage(
                question, history_messages, model_id, history_text
            ),
        }


class DirectAnswer(Runnable[State, dict[str, object]]):
    """
    LLM with only user message (no RAG, no tools). Writes direct_answer.

    Best practices:
    - Simple LLM invocation without tools
    - Handles empty input gracefully
    """

    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Sync direct answer generation."""
        log_node_start("DirectAnswer")
        ensure_required(input, "DirectAnswer")
        t0 = time.perf_counter()
        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        model_id = configurable.get("model_id")

        question = input.get("user_request", "")
        history_messages = input.get("messages", [])
        history_text = cast(str | None, input.get("history_text"))
        usage_snapshot = _compute_context_usage(question, history_messages, model_id, history_text)
        if not question.strip():
            logger.debug("DirectAnswer: Empty question, returning empty answer")
            log_node_end("DirectAnswer", next="DraftAnswer", answer_len=0)
            return {"direct_answer": "", "context_usage": usage_snapshot}

        try:
            llm = get_llm(model_id=model_id)

            # Build prompt from prior messages, avoiding duplicate of current user question
            prompt_messages = _ensure_message_list(history_messages)

            append_question = True
            # Find the last user message (if any) and compare with current question
            for m in reversed(prompt_messages):
                if isinstance(m, HumanMessage):
                    if (
                        question
                        and isinstance(m.content, str)
                        and m.content.strip() == question.strip()
                    ):
                        append_question = False
                    break

            if append_question and question:
                prompt_messages.append(HumanMessage(content=question))

            # Fallback: if nothing usable, at least include the question
            if not prompt_messages and question:
                prompt_messages = [HumanMessage(content=question)]

            msg = llm.invoke(prompt_messages, config=run_config)
            raw_content = getattr(msg, "content", "")
            if isinstance(raw_content, str):
                content = raw_content.strip()
            else:
                content = str(raw_content) if raw_content is not None else ""
            logger.debug("DirectAnswer: Generated answer_len=%d", len(content))
            duration_ms = (time.perf_counter() - t0) * 1000
            log_node_end(
                "DirectAnswer", duration_ms=duration_ms, next="DraftAnswer", answer_len=len(content)
            )
            return {"direct_answer": content, "context_usage": usage_snapshot}
        except Exception as e:
            logger.error("DirectAnswer error: %s", e, exc_info=True)
            log_node_end(
                "DirectAnswer", duration_ms=(time.perf_counter() - t0) * 1000, error=str(e)
            )
            return {
                "direct_answer": "",
                "error": f"Direct answer generation failed: {str(e)}",
                "context_usage": usage_snapshot,
            }

    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Async direct answer generation (runs sync LLM in thread for streaming path)."""
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)

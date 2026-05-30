"""ChatRuntimeService: runtime boundary between FastAPI and OCI-backed chat execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool

from api.settings import get_settings
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config
from src.rag_agent.utils.langfuse_tracing import (
    LangfuseChatTrace,
    add_langfuse_callbacks,
    start_langfuse_chat_trace,
)

from . import rag_runtime
from .llm_invocation import invoke_llm_with_optional_config
from .mcp_turn import run_mcp_agent_turn, tool_failure_summary
from .memory import (
    chat_history_before_latest_user,
    contextualize_question,
    hydrate_thread_messages,
    latest_user_message,
    new_incoming_messages,
    to_langchain_messages,
)
from .observability import emit_usage_observability, extract_usage
from .streaming import v3_raw_event
from .thread_checkpoints import LangGraphCheckpointThreadStateStore

logger = logging.getLogger(__name__)

_ORACLE_RETRIEVAL_TOOL_NAME = "oracle_retrieval"
_NO_ORACLE_CONTEXT_ANSWER = "I don't know the answer from the selected Oracle collection."
_ORACLE_RETRIEVAL_FAILED_ANSWER = (
    "I couldn't retrieve context from the selected Oracle collection because retrieval failed. "
    "Please try again after the database is available."
)


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


def _build_run_config(
    *,
    thread_id: str | None,
    mcp_server_keys: list[str] | None,
) -> RunnableConfig | None:
    configurable: dict[str, object] = {}
    if thread_id:
        configurable["thread_id"] = thread_id
    if mcp_server_keys:
        configurable["mcp_server_keys"] = mcp_server_keys
    if not configurable:
        return None
    return cast(RunnableConfig, {"configurable": configurable})


def _prepare_run_config(
    *,
    thread_id: str | None,
    mcp_server_keys: list[str] | None,
    mode: str | None,
    model_id: str | None,
    session_id: str | None,
    enable_tracing: bool | None,
    trace_context: dict[str, str] | None = None,
) -> RunnableConfig:
    base = (
        _build_run_config(
            thread_id=thread_id,
            mcp_server_keys=mcp_server_keys,
        )
        or {}
    )
    run_config: dict[str, object] = dict(base)
    configurable = run_config.get("configurable")
    if isinstance(configurable, dict):
        run_config["configurable"] = dict(configurable)
    if enable_tracing is not True:
        return cast(RunnableConfig, run_config)
    configurable_payload = cast(dict[str, object], run_config.get("configurable") or {})
    if mode:
        configurable_payload["mode"] = mode
    if model_id:
        configurable_payload["model_id"] = model_id
    if configurable_payload:
        run_config["configurable"] = configurable_payload
    add_langfuse_callbacks(
        run_config,
        session_id=session_id,
        user_id=None,
        trace_context=trace_context,
        trace_name=f"chat-{mode or 'unknown'}",
        tags=[
            tag
            for tag in (
                "chat",
                f"mode:{mode}" if mode else None,
                f"model:{model_id}" if model_id else None,
            )
            if tag is not None
        ],
    )
    return cast(RunnableConfig, run_config)


def _resolve_effective_mode(mode: str | None) -> str:
    explicit = str(mode or "").strip().lower()
    if explicit in {"direct", "rag", "mcp", "mixed"}:
        return explicit
    settings = get_settings()
    enable_mcp_tools = bool(getattr(settings, "ENABLE_MCP_TOOLS", True))
    mcp_config = get_mcp_servers_config()
    if enable_mcp_tools and bool(mcp_config):
        return "mixed"
    return "rag"


def _to_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _workflow_policy_for_request(*, mode: str, question: str) -> dict[str, object] | None:
    policy_raw = getattr(get_settings(), "MCP_WORKFLOW_POLICY", {})
    if not isinstance(policy_raw, dict) or not policy_raw:
        return None
    enabled = bool(policy_raw.get("enabled", True))
    if not enabled:
        return None
    apply_modes = _to_string_list(policy_raw.get("apply_modes")) or ["mixed"]
    if mode not in {m.lower() for m in apply_modes}:
        return None
    activation_terms = [
        term.lower() for term in _to_string_list(policy_raw.get("activation_terms"))
    ]
    lower_question = question.strip().lower()
    if activation_terms and not any(term in lower_question for term in activation_terms):
        return None
    required_capabilities = _to_string_list(policy_raw.get("required_capabilities"))
    tool_capability_map_raw = policy_raw.get("tool_capability_map")
    if not required_capabilities or not isinstance(tool_capability_map_raw, dict):
        return None
    tool_capability_map: dict[str, list[str]] = {}
    for tool_name, capabilities in tool_capability_map_raw.items():
        normalized_tool_name = str(tool_name).strip().lower()
        if not normalized_tool_name:
            continue
        caps = _to_string_list(capabilities)
        if caps:
            tool_capability_map[normalized_tool_name] = [cap.lower() for cap in caps]
    if not tool_capability_map:
        return None
    return {
        "required_capabilities": [cap.lower() for cap in required_capabilities],
        "tool_capability_map": tool_capability_map,
        "failure_message": str(policy_raw.get("failure_message") or "").strip(),
    }


def _require_tool_call_enabled() -> bool:
    return bool(getattr(get_settings(), "REQUIRE_TOOL_CALL", False))


def _repeated_workflow_controller_enabled() -> bool:
    return bool(getattr(get_settings(), "MCP_REPEATED_WORKFLOW_CONTROLLER", False))


def _workflow_checkpoint_path() -> str | None:
    settings = get_settings()
    if not bool(getattr(settings, "ENABLE_PERSISTENT_MEMORY", False)):
        return None
    raw_path = str(getattr(settings, "LANGGRAPH_SQLITE_PATH", "") or "").strip()
    return raw_path or None


def _enforce_workflow_policy(
    *,
    policy: dict[str, object] | None,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> tuple[bool, list[str], str | None]:
    if policy is None:
        return False, [], None
    required_capabilities = _to_string_list(policy.get("required_capabilities"))
    tool_capability_map = cast(dict[str, list[str]], policy.get("tool_capability_map") or {})
    if not required_capabilities or not tool_capability_map:
        return False, [], None
    called_capabilities: set[str] = set()
    for tool_name in _called_tool_names(
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    ):
        for capability in tool_capability_map.get(tool_name, []):
            called_capabilities.add(capability.lower())
    missing = [cap for cap in required_capabilities if cap.lower() not in called_capabilities]
    if not missing:
        return True, [], None
    default_message = (
        "Workflow validation failed. Missing required steps: "
        + ", ".join(missing)
        + ". Please continue with the required workflow tools."
    )
    failure_message = str(policy.get("failure_message") or "").strip() or default_message
    return True, missing, failure_message


def _is_trivial_answer(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    return not any(ch.isalnum() for ch in stripped)


def _called_tool_names(
    *,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> set[str]:
    names = {str(name).strip().lower() for name in tools_used if str(name).strip()}
    names.update(
        str(invocation.get("tool_name") or "").strip().lower()
        for invocation in tool_invocations
        if isinstance(invocation, dict)
        and str(invocation.get("tool_name") or "").strip()
    )
    return names


def _tool_was_called(
    *,
    tool_name: str,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> bool:
    expected = tool_name.strip().lower()
    return expected in _called_tool_names(
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    )


def _references_from_result(
    result: dict[str, object],
    *,
    include_empty_core: bool = False,
    include_empty_mcp_tools: bool = False,
) -> dict[str, object]:
    references: dict[str, object] = {}
    standalone = result.get("standalone_question")
    if include_empty_core:
        references["standalone_question"] = standalone if isinstance(standalone, str) else None
    elif isinstance(standalone, str) and standalone.strip():
        references["standalone_question"] = standalone.strip()

    citations = result.get("citations")
    if isinstance(citations, list):
        references["citations"] = citations
    elif include_empty_core:
        references["citations"] = []

    reranker_docs = result.get("reranker_docs")
    if isinstance(reranker_docs, list):
        references["reranker_docs"] = reranker_docs
    elif include_empty_core:
        references["reranker_docs"] = []

    context_usage = result.get("context_usage")
    if isinstance(context_usage, dict):
        references["context_usage"] = context_usage

    trace_id = result.get("trace_id")
    if isinstance(trace_id, str) and trace_id.strip():
        references["trace_id"] = trace_id.strip()

    if result.get("mcp_used") is True:
        references["mcp_used"] = True

    mcp_tools_used = result.get("mcp_tools_used")
    if isinstance(mcp_tools_used, list):
        tool_names = [str(tool) for tool in mcp_tools_used if str(tool).strip()]
        if tool_names or include_empty_mcp_tools:
            references["mcp_tools_used"] = tool_names

    mcp_invocations = result.get("mcp_tool_invocations")
    if isinstance(mcp_invocations, list) and mcp_invocations:
        references["mcp_tool_invocations"] = mcp_invocations

    error_value = result.get("error")
    if isinstance(error_value, str) and error_value.strip():
        references["error"] = error_value.strip()

    return references


def _oracle_retrieval_used_without_context(
    *,
    retrieval_state: object,
    retrieval_docs: list[Document],
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> bool:
    if retrieval_docs:
        return False
    if not isinstance(retrieval_state, dict):
        return False
    if str(retrieval_state.get("error") or "").strip():
        return False
    return _tool_was_called(
        tool_name=_ORACLE_RETRIEVAL_TOOL_NAME,
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    )


def _oracle_retrieval_error(
    *,
    retrieval_state: object,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> str | None:
    if not isinstance(retrieval_state, dict):
        return None
    error = str(retrieval_state.get("error") or "").strip()
    if not error:
        return None
    if not _tool_was_called(
        tool_name=_ORACLE_RETRIEVAL_TOOL_NAME,
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    ):
        return None
    return error


def _mixed_tool_supplemental_context(
    tool_invocations: list[dict[str, object]],
) -> str | None:
    blocks: list[str] = []
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool_name") or "").strip()
        if not tool_name or tool_name == _ORACLE_RETRIEVAL_TOOL_NAME:
            continue
        error = str(invocation.get("error") or "").strip()
        result = str(invocation.get("result") or "").strip()
        if error:
            result = f"Error: {error}"
        if not result:
            continue
        args = invocation.get("args")
        args_text = f"\nArgs: {args}" if args not in (None, {}, []) else ""
        blocks.append(f"Tool: {tool_name}{args_text}\nResult: {result}")
    return "\n\n".join(blocks) or None


class ChatRuntimeService:
    """Small service to execute direct, MCP, RAG, and mixed OCI chat modes."""

    def __init__(
        self,
        graph: Any = None,
        thread_state_store: LangGraphCheckpointThreadStateStore | None = None,
    ) -> None:
        _ = graph
        self._thread_state: dict[str, dict[str, Any]] = {}
        self._thread_state_store = thread_state_store

    async def run_chat(
        self,
        *,
        messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        collection_name: str | None,
        enable_reranker: bool | None,
        enable_tracing: bool | None,
        mode: str | None,
        mcp_server_keys: list[str] | None,
        stream: bool,
        tool_progress_callback: Callable[[dict[str, object]], None] | None = None,
        answer_delta_callback: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        normalized_mode = _resolve_effective_mode(mode)
        question = latest_user_message(messages)
        with start_langfuse_chat_trace(
            enabled=enable_tracing,
            mode=normalized_mode,
            model_id=model_id,
            session_id=session_id,
            thread_id=thread_id,
            input_payload={"question": question} if question else None,
        ) as langfuse_trace:
            result = await self._run_chat_impl(
                messages=messages,
                model_id=model_id,
                thread_id=thread_id,
                session_id=session_id,
                collection_name=collection_name,
                enable_reranker=enable_reranker,
                enable_tracing=enable_tracing,
                mode=mode,
                mcp_server_keys=mcp_server_keys,
                stream=stream,
                tool_progress_callback=tool_progress_callback,
                answer_delta_callback=answer_delta_callback,
                langfuse_trace=langfuse_trace,
            )
            if langfuse_trace.trace_id:
                result["trace_id"] = langfuse_trace.trace_id
            langfuse_trace.update_output(
                {
                    "has_error": bool(result.get("error")),
                    "mcp_used": bool(result.get("mcp_used")),
                    "tools_used": result.get("mcp_tools_used") or [],
                }
            )
            return result

    async def _run_chat_impl(
        self,
        *,
        messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        collection_name: str | None,
        enable_reranker: bool | None,
        enable_tracing: bool | None,
        mode: str | None,
        mcp_server_keys: list[str] | None,
        stream: bool,
        tool_progress_callback: Callable[[dict[str, object]], None] | None = None,
        answer_delta_callback: Callable[[str], None] | None = None,
        langfuse_trace: LangfuseChatTrace | None = None,
    ) -> dict[str, object]:
        normalized_mode = _resolve_effective_mode(mode)
        incoming_messages = messages
        conversation_messages = self._hydrate_thread_messages(thread_id, incoming_messages)

        if normalized_mode == "mixed":
            latest_user_message = self._latest_user_message(conversation_messages)
            chat_history = self._chat_history_before_latest_user(conversation_messages)

            retrieval_tool = self._build_oracle_retrieval_tool(collection_name)
            resolved_model_id = model_id or get_llm().model_id
            run_cfg = _prepare_run_config(
                thread_id=thread_id,
                mcp_server_keys=mcp_server_keys,
                mode=normalized_mode,
                model_id=resolved_model_id,
                session_id=session_id,
                enable_tracing=enable_tracing,
                trace_context=langfuse_trace.trace_context if langfuse_trace else None,
            )
            mcp_turn = await run_mcp_agent_turn(
                question=latest_user_message,
                chat_history=chat_history,
                resolved_model_id=resolved_model_id,
                run_config=run_cfg,
                mode=normalized_mode,
                mcp_server_keys=mcp_server_keys,
                require_tool_call=_require_tool_call_enabled(),
                repeated_workflow_enabled=_repeated_workflow_controller_enabled(),
                workflow_checkpoint_path=_workflow_checkpoint_path(),
                tool_progress_callback=tool_progress_callback,
                answer_delta_callback=None,
                stop_after_tool_names=None,
                extra_tools=[retrieval_tool],
                require_mcp_tool_call_when_referenced=True,
            )
            final_answer = mcp_turn.answer
            tools_used = mcp_turn.tools_used
            tool_invocations = mcp_turn.tool_invocations
            workflow_policy = _workflow_policy_for_request(
                mode=normalized_mode,
                question=latest_user_message,
            )
            policy_applied, missing_capabilities, policy_failure_message = _enforce_workflow_policy(
                policy=workflow_policy,
                tools_used=tools_used,
                tool_invocations=cast(list[dict[str, object]], tool_invocations),
            )
            policy_error = (
                policy_failure_message if policy_applied and missing_capabilities else None
            )
            if policy_error:
                final_answer = policy_error
            tool_failure_error = tool_failure_summary(
                cast(list[dict[str, object]], tool_invocations)
            )
            if not policy_error and _is_trivial_answer(final_answer) and tool_failure_error:
                final_answer = tool_failure_error
                policy_error = tool_failure_error
            retrieval_state = getattr(retrieval_tool, "_retrieval_state", None)
            retrieval_docs = (
                cast(list[Document], retrieval_state.get("docs", []))
                if isinstance(retrieval_state, dict)
                else []
            )
            retrieval_error = _oracle_retrieval_error(
                retrieval_state=retrieval_state,
                tools_used=tools_used,
                tool_invocations=cast(list[dict[str, object]], tool_invocations),
            )
            if retrieval_docs and latest_user_message:
                retrieval_docs = rag_runtime.rerank_retrieved_docs(
                    latest_user_message,
                    retrieval_docs,
                    enable_reranker=enable_reranker,
                )
            if not policy_error and retrieval_error:
                final_answer = _ORACLE_RETRIEVAL_FAILED_ANSWER
                policy_error = _ORACLE_RETRIEVAL_FAILED_ANSWER
            if (
                not policy_error
                and _oracle_retrieval_used_without_context(
                    retrieval_state=retrieval_state,
                    retrieval_docs=retrieval_docs,
                    tools_used=tools_used,
                    tool_invocations=cast(list[dict[str, object]], tool_invocations),
                )
            ):
                final_answer = _NO_ORACLE_CONTEXT_ANSWER
            if not policy_error and retrieval_docs:
                supplemental_context = _mixed_tool_supplemental_context(
                    cast(list[dict[str, object]], tool_invocations)
                )
                if stream and answer_delta_callback is not None:
                    answer_parts: list[str] = []
                    async for text_delta, _chunk, stream_model_id in rag_runtime.stream_rag_answer(
                        question=latest_user_message,
                        docs=retrieval_docs,
                        model_id=model_id,
                        run_config=run_cfg,
                        supplemental_context=supplemental_context,
                    ):
                        if isinstance(stream_model_id, str) and stream_model_id.strip():
                            resolved_model_id = stream_model_id
                        if text_delta:
                            answer_parts.append(text_delta)
                            answer_delta_callback(text_delta)
                    final_answer = "".join(answer_parts).strip()
                else:
                    final_answer, _rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(
                        question=latest_user_message,
                        docs=retrieval_docs,
                        model_id=model_id,
                        run_config=run_cfg,
                        supplemental_context=supplemental_context,
                    )
            mixed_result: dict[str, object] = {
                "final_answer": final_answer,
                "error": policy_error,
                "standalone_question": latest_user_message or None,
                "citations": rag_runtime.citations_from_docs(retrieval_docs),
                "reranker_docs": rag_runtime.serialize_docs(retrieval_docs),
                "context_usage": (
                    {"retrieved_docs_count": len(retrieval_docs)} if retrieval_docs else None
                ),
                "mcp_used": bool(tools_used),
                "mcp_tools_used": tools_used,
                "mcp_tool_invocations": tool_invocations,
            }
            if isinstance(resolved_model_id, str) and resolved_model_id.strip():
                mixed_result["model_id"] = resolved_model_id.strip()
            self._attach_trace_id(mixed_result, langfuse_trace)
            self._store_thread_state(thread_id, incoming_messages, mixed_result)
            return mixed_result
        if normalized_mode != "direct":
            if normalized_mode == "mcp":
                question = self._latest_user_message(conversation_messages)
                chat_history = self._chat_history_before_latest_user(conversation_messages)

                resolved_model_id = model_id or get_llm().model_id
                run_cfg = _prepare_run_config(
                    thread_id=thread_id,
                    mcp_server_keys=mcp_server_keys,
                    mode=normalized_mode,
                    model_id=resolved_model_id,
                    session_id=session_id,
                    enable_tracing=enable_tracing,
                    trace_context=langfuse_trace.trace_context if langfuse_trace else None,
                )
                mcp_turn = await run_mcp_agent_turn(
                    question=question,
                    chat_history=chat_history,
                    resolved_model_id=resolved_model_id,
                    run_config=run_cfg,
                    mode=normalized_mode,
                    mcp_server_keys=mcp_server_keys,
                    require_tool_call=_require_tool_call_enabled(),
                    repeated_workflow_enabled=_repeated_workflow_controller_enabled(),
                    workflow_checkpoint_path=_workflow_checkpoint_path(),
                    tool_progress_callback=tool_progress_callback,
                    answer_delta_callback=answer_delta_callback,
                )
                mcp_result: dict[str, object] = {
                    "final_answer": mcp_turn.answer,
                    "error": None,
                    "standalone_question": question or None,
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": None,
                    "mcp_used": bool(mcp_turn.tools_used),
                    "mcp_tools_used": mcp_turn.tools_used,
                    "mcp_tool_invocations": mcp_turn.tool_invocations,
                }
                tool_failure_error = tool_failure_summary(
                    mcp_turn.tool_invocations
                )
                if (
                    _is_trivial_answer(str(mcp_result.get("final_answer") or ""))
                    and tool_failure_error
                ):
                    mcp_result["final_answer"] = tool_failure_error
                    mcp_result["error"] = tool_failure_error
                mcp_result["model_id"] = mcp_turn.resolved_model_id
                self._attach_trace_id(mcp_result, langfuse_trace)
                self._store_thread_state(thread_id, incoming_messages, mcp_result)
                return mcp_result
            if normalized_mode == "rag":
                question = self._latest_user_message(conversation_messages)
                chat_history = self._chat_history_before_latest_user(conversation_messages)
                run_cfg = _prepare_run_config(
                    thread_id=thread_id,
                    mcp_server_keys=mcp_server_keys,
                    mode=normalized_mode,
                    model_id=model_id,
                    session_id=session_id,
                    enable_tracing=enable_tracing,
                    trace_context=langfuse_trace.trace_context if langfuse_trace else None,
                )
                standalone_question = await self._contextualize_question(
                    question=question,
                    chat_history=chat_history,
                    model_id=model_id,
                    run_config=run_cfg,
                )

                docs = rag_runtime.retrieve_oracle_docs(
                    query=standalone_question,
                    collection_name=collection_name,
                    k=5,
                )
                docs = rag_runtime.rerank_retrieved_docs(
                    standalone_question,
                    docs,
                    enable_reranker=enable_reranker,
                )
                if docs and stream and answer_delta_callback is not None:
                    answer_parts: list[str] = []
                    last_chunk: object | None = None
                    resolved_model_id = model_id or "unknown"
                    async for text_delta, chunk, stream_model_id in rag_runtime.stream_rag_answer(
                        question=standalone_question,
                        docs=docs,
                        model_id=model_id,
                        run_config=run_cfg,
                    ):
                        resolved_model_id = stream_model_id
                        last_chunk = chunk
                        if text_delta:
                            answer_parts.append(text_delta)
                            answer_delta_callback(text_delta)
                    rag_answer = "".join(answer_parts).strip()
                    rag_usage = extract_usage(last_chunk) if last_chunk is not None else None
                elif docs:
                    rag_answer, rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(
                        question=standalone_question,
                        docs=docs,
                        model_id=model_id,
                        run_config=run_cfg,
                    )
                else:
                    rag_answer = _NO_ORACLE_CONTEXT_ANSWER
                    rag_usage = None
                    resolved_model_id = model_id or "unknown"
                emitted_usage, cost_usd = emit_usage_observability(
                    mode=normalized_mode,
                    model_id=resolved_model_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    usage=rag_usage,
                )
                rag_result: dict[str, object] = {
                    "final_answer": rag_answer,
                    "error": None,
                    "standalone_question": standalone_question or None,
                    "citations": rag_runtime.citations_from_docs(docs),
                    "reranker_docs": rag_runtime.serialize_docs(docs),
                    "context_usage": None,
                    "mcp_used": False,
                    "mcp_tools_used": [],
                    "model_id": resolved_model_id,
                    "usage": emitted_usage,
                    "cost_usd": cost_usd,
                }
                self._attach_trace_id(rag_result, langfuse_trace)
                self._store_thread_state(thread_id, incoming_messages, rag_result)
                return rag_result
            raise NotImplementedError(
                "run_chat currently only handles direct, mcp, rag, and mixed modes"
            )

        history: list[Any] = []
        latest_user_message = ""
        for item in conversation_messages:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "user":
                history.append(HumanMessage(content=content))
                latest_user_message = content.strip() or latest_user_message
            elif role == "assistant":
                history.append(AIMessage(content=content))

        run_cfg = _prepare_run_config(
            thread_id=thread_id,
            mcp_server_keys=mcp_server_keys,
            mode=normalized_mode,
            model_id=model_id,
            session_id=session_id,
            enable_tracing=enable_tracing,
            trace_context=langfuse_trace.trace_context if langfuse_trace else None,
        )
        llm = get_llm(model_id=model_id)
        response = await asyncio.to_thread(invoke_llm_with_optional_config, llm, history, run_cfg)
        usage = extract_usage(response)
        resolved_model_id = cast(str | None, getattr(llm, "model_id", None)) or model_id
        emitted_usage, cost_usd = emit_usage_observability(
            mode=normalized_mode,
            model_id=resolved_model_id,
            session_id=session_id,
            thread_id=thread_id,
            usage=usage,
        )
        final_answer = str(getattr(response, "content", "") or "").strip()
        direct_result: dict[str, object] = {
            "final_answer": final_answer,
            "error": None,
            "standalone_question": latest_user_message or None,
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "model_id": resolved_model_id,
            "usage": emitted_usage,
            "cost_usd": cost_usd,
        }
        self._attach_trace_id(direct_result, langfuse_trace)
        self._store_thread_state(thread_id, incoming_messages, direct_result)
        return direct_result

    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object] | None = None,
        version: str = "v3",
    ) -> AsyncIterator[dict[str, object]]:
        if version != "v3":
            raise ValueError("ChatRuntimeService only supports stream_events version='v3'.")

        raw_messages = input_payload.get("messages")
        messages = cast(
            list[dict[str, object]], raw_messages if isinstance(raw_messages, list) else []
        )
        configurable = config.get("configurable") if isinstance(config, dict) else None
        cfg = cast(dict[str, object], configurable if isinstance(configurable, dict) else {})

        result: dict[str, object] = {}
        error: BaseException | None = None
        event_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        run_done = asyncio.Event()
        loop = asyncio.get_running_loop()
        answer_delta_emitted = False

        def _emit_tool_progress(payload: dict[str, object]) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, ("tool_calls", payload))

        def _emit_answer_delta(delta: str) -> None:
            nonlocal answer_delta_emitted
            text = str(delta or "")
            if not text:
                return
            answer_delta_emitted = True
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                (
                    "messages",
                    (
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": text},
                        },
                        {"langgraph_node": "chat_runtime"},
                    ),
                ),
            )

        async def _run() -> None:
            nonlocal result, error
            try:
                result = await self.run_chat(
                    messages=messages,
                    model_id=cast(str | None, cfg.get("model_id")),
                    thread_id=cast(str | None, cfg.get("thread_id")),
                    session_id=cast(str | None, cfg.get("session_id")),
                    collection_name=cast(str | None, cfg.get("collection_name")),
                    enable_reranker=cast(bool | None, cfg.get("enable_reranker")),
                    enable_tracing=cast(bool | None, cfg.get("enable_tracing")),
                    mode=cast(str | None, cfg.get("mode")),
                    mcp_server_keys=cast(list[str] | None, cfg.get("mcp_server_keys")),
                    stream=True,
                    tool_progress_callback=_emit_tool_progress,
                    answer_delta_callback=_emit_answer_delta,
                )
            except BaseException as exc:  # noqa: BLE001
                error = exc
            finally:
                run_done.set()

        run_task = asyncio.create_task(_run())
        while True:
            if run_done.is_set() and event_queue.empty():
                break
            try:
                method, data = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            yield v3_raw_event(method=method, data=data)

        await run_task
        if error is not None:
            raise error

        answer = str(result.get("final_answer") or "")
        if answer and not answer_delta_emitted:
            yield v3_raw_event(
                method="messages",
                data=(
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": answer},
                    },
                    {"langgraph_node": "chat_runtime"},
                ),
            )

        yield v3_raw_event(
            method="custom",
            data={
                "type": "references",
                "data": _references_from_result(result, include_empty_core=True),
            },
        )

    def _build_oracle_retrieval_tool(self, collection_name: str | None) -> StructuredTool:
        return rag_runtime.build_oracle_retrieval_tool(
            collection_name=collection_name,
            filter_docs=rag_runtime.filter_retrieved_docs,
        )

    def _attach_trace_id(
        self,
        result: dict[str, object],
        langfuse_trace: LangfuseChatTrace | None,
    ) -> None:
        if langfuse_trace is not None and langfuse_trace.trace_id:
            result["trace_id"] = langfuse_trace.trace_id

    def _hydrate_thread_messages(
        self,
        thread_id: str | None,
        incoming_messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not thread_id:
            return list(incoming_messages)
        return hydrate_thread_messages(self._get_thread_state(thread_id), incoming_messages)

    def _latest_user_message(self, messages: list[dict[str, object]]) -> str:
        return latest_user_message(messages)

    def _chat_history_before_latest_user(self, messages: list[dict[str, object]]) -> list[Any]:
        return chat_history_before_latest_user(messages)

    async def _contextualize_question(
        self,
        *,
        question: str,
        chat_history: list[Any],
        model_id: str | None,
        run_config: RunnableConfig | None,
    ) -> str:
        return await contextualize_question(
            question=question,
            chat_history=chat_history,
            model_id=model_id,
            run_config=run_config,
        )

    def _new_incoming_messages(
        self,
        prior_messages: list[object],
        incoming_messages: list[object],
    ) -> list[Any]:
        return new_incoming_messages(prior_messages, incoming_messages)

    async def get_state(self, run_config: dict[str, Any]) -> Any:
        thread_id = self._thread_id_from_run_config(run_config)
        values = self._get_thread_state(thread_id) if thread_id else {}
        return type("StateSnapshot", (), {"values": values})()

    def get_state_values(self, run_config: dict[str, Any]) -> dict[str, Any] | None:
        thread_id = self._thread_id_from_run_config(run_config)
        if not thread_id:
            return None
        return self._get_thread_state(thread_id)

    async def delete_thread(self, thread_id: str) -> None:
        self._thread_state.pop(thread_id, None)
        if self._thread_state_store is not None:
            self._thread_state_store.delete(thread_id)

    def _thread_id_from_run_config(self, run_config: dict[str, Any]) -> str | None:
        configurable = run_config.get("configurable")
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
        thread_id = run_config.get("thread_id")
        if isinstance(thread_id, str) and thread_id.strip():
            return thread_id.strip()
        return None

    def _store_thread_state(
        self,
        thread_id: str | None,
        messages: list[dict[str, object]],
        result: dict[str, Any],
    ) -> None:
        if not thread_id:
            return

        prior_messages = list((self._get_thread_state(thread_id) or {}).get("messages") or [])
        incoming_messages = self._to_langchain_messages(messages)
        updated_messages = prior_messages + self._new_incoming_messages(
            prior_messages,
            incoming_messages,
        )
        final_answer = str(result.get("final_answer") or "").strip()
        if final_answer:
            references = _references_from_result(
                result,
                include_empty_mcp_tools=True,
            )
            updated_messages.append(
                AIMessage(
                    content=final_answer,
                    additional_kwargs=references or {},
                    response_metadata=references or {},
                )
            )

        state = {
            "messages": updated_messages,
            **result,
        }
        self._thread_state[thread_id] = state
        if self._thread_state_store is not None:
            self._thread_state_store.put(thread_id, state)

    def _to_langchain_messages(self, messages: list[dict[str, object]]) -> list[Any]:
        return to_langchain_messages(messages)

    def _get_thread_state(self, thread_id: str | None) -> dict[str, Any]:
        if not thread_id:
            return {}
        if thread_id in self._thread_state:
            return self._thread_state[thread_id]
        if self._thread_state_store is None:
            return {}
        state = self._thread_state_store.get(thread_id)
        if state is None:
            return {}
        self._thread_state[thread_id] = state
        return state

"""ChatRuntimeService: runtime boundary between FastAPI and OCI-backed chat execution."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from api.settings import get_settings
from src.rag_agent.infrastructure import db_utils as _db_utils
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.infrastructure import retrieval as _retrieval
from src.rag_agent.infrastructure.direct_mcp_tools import get_mcp_tools_async
from src.rag_agent.infrastructure.mcp_agent import get_mcp_answer_async
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config
from src.rag_agent.utils.langfuse_tracing import (
    LangfuseChatTrace,
    add_langfuse_callbacks,
    start_langfuse_chat_trace,
)
from src.rag_agent.workflows.mcp_repeated import run_repeated_mcp_workflow
from src.rag_agent.workflows.workflow_intent import should_use_repeated_workflow

from . import rag_runtime
from .llm_invocation import invoke_llm_with_optional_config
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


def get_pooled_connection() -> Any:
    return _db_utils.get_pooled_connection()


def get_embedding_model() -> Any:
    return _oci_models.get_embedding_model()


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


get_oracle_vs = _oci_models.get_oracle_vs


def search_documents(**kwargs: object) -> list[Document]:
    return cast(list[Document], _retrieval.search_documents(**cast(Any, kwargs)))


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    return _oci_models.rerank_documents(query, docs)


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


def _question_explicitly_references_mcp_tools(
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


def _has_called_mcp_tool(
    tools_used: list[str],
    mcp_tools: list[BaseTool],
) -> bool:
    used = {str(name).strip() for name in tools_used if str(name).strip()}
    mcp_tool_names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in mcp_tools
        if str(getattr(tool, "name", "") or "").strip()
    }
    return any(name in used for name in mcp_tool_names)


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
    called_tool_names = {str(name).strip().lower() for name in tools_used if str(name).strip()}
    called_tool_names.update(
        str(inv.get("tool_name") or "").strip().lower()
        for inv in tool_invocations
        if isinstance(inv, dict)
    )
    called_capabilities: set[str] = set()
    for tool_name in called_tool_names:
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


def _tool_failure_summary(tool_invocations: list[dict[str, object]]) -> str | None:
    failed_tools: list[str] = []
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool_name") or "").strip()
        result_text = str(invocation.get("result") or "").strip()
        if not result_text:
            continue
        lowered = result_text.lower()
        if "failed after" in lowered or "toolexception" in lowered or "error" in lowered:
            if tool_name and tool_name not in failed_tools:
                failed_tools.append(tool_name)
    if not failed_tools:
        return None
    joined = ", ".join(failed_tools)
    return f"Workflow failed because tool execution failed: {joined}. See tool output for details."


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
            try:
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
                    langfuse_trace=langfuse_trace,
                )
            except BaseException:
                raise
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
        langfuse_trace: LangfuseChatTrace | None = None,
    ) -> dict[str, object]:
        _ = (
            session_id,
            collection_name,
            enable_reranker,
            enable_tracing,
            mcp_server_keys,
            stream,
        )
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
            tool_load_started = time.perf_counter()
            mcp_tools = await get_mcp_tools_async(
                server_keys=mcp_server_keys,
                run_config=run_cfg,
            )
            logger.info(
                "chat_runtime_mcp_tools_loaded mode=%s tool_count=%d elapsed_ms=%.1f",
                normalized_mode,
                len(mcp_tools),
                (time.perf_counter() - tool_load_started) * 1000,
            )
            all_tools = [retrieval_tool, *mcp_tools]
            repeated_result = None
            if _repeated_workflow_controller_enabled() and await should_use_repeated_workflow(
                question=latest_user_message,
                tools=all_tools,
                model_id=resolved_model_id,
                run_config=run_cfg,
            ):
                repeated_result = await run_repeated_mcp_workflow(
                    question=latest_user_message,
                    model_id=resolved_model_id,
                    tools=all_tools,
                    run_config=run_cfg,
                    require_tool_call=_require_tool_call_enabled(),
                    get_answer=get_mcp_answer_async,
                    checkpoint_path=_workflow_checkpoint_path(),
                    tool_progress_callback=tool_progress_callback,
                    chat_history=chat_history,
                )
            if repeated_result is None:
                final_answer, tools_used, tool_invocations = await get_mcp_answer_async(
                    latest_user_message,
                    chat_history=chat_history,
                    model_id=resolved_model_id,
                    tools=all_tools,
                    run_config=run_cfg,
                    require_tool_call=_require_tool_call_enabled(),
                    tool_progress_callback=tool_progress_callback,
                )
            else:
                final_answer, tools_used, tool_invocations = repeated_result
            explicit_mcp_required = _question_explicitly_references_mcp_tools(
                latest_user_message,
                mcp_tools,
            )
            workflow_policy = _workflow_policy_for_request(
                mode=normalized_mode,
                question=latest_user_message,
            )
            if (
                explicit_mcp_required
                and latest_user_message
                and not _has_called_mcp_tool(tools_used, mcp_tools)
            ):
                final_answer, tools_used, tool_invocations = await get_mcp_answer_async(
                    latest_user_message,
                    chat_history=chat_history,
                    model_id=resolved_model_id,
                    tools=[retrieval_tool, *mcp_tools],
                    run_config=run_cfg,
                    require_tool_call=True,
                    tool_progress_callback=tool_progress_callback,
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
            tool_failure_error = _tool_failure_summary(
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
            if not retrieval_docs and "oracle_retrieval" in tools_used and latest_user_message:
                retrieval_docs = self._retrieve_oracle_docs(
                    query=latest_user_message,
                    collection_name=collection_name,
                    k=8,
                )
            if retrieval_docs and latest_user_message:
                retrieval_docs = self._rank_retrieved_docs(
                    latest_user_message,
                    retrieval_docs,
                    enable_reranker=enable_reranker,
                )
            # Guardrail: if no MCP tools were used at all, fall back to direct
            # RAG retrieval+synthesis so doc-grounded questions don't bypass DB.
            # Do not override successful non-retrieval MCP tool answers (e.g. calculator).
            if (
                latest_user_message
                and not tools_used
                and not explicit_mcp_required
                and not policy_applied
            ):
                retrieval_docs = self._retrieve_oracle_docs(
                    query=latest_user_message,
                    collection_name=collection_name,
                    k=8,
                )
                if retrieval_docs:
                    retrieval_docs = self._rank_retrieved_docs(
                        latest_user_message,
                        retrieval_docs,
                        enable_reranker=enable_reranker,
                    )
                    rag_answer, rag_usage, resolved_model_id = await self._synthesize_rag_answer(
                        question=latest_user_message,
                        docs=retrieval_docs,
                        model_id=model_id,
                    )
                    emit_usage_observability(
                        mode=normalized_mode,
                        model_id=resolved_model_id,
                        session_id=session_id,
                        thread_id=thread_id,
                        usage=rag_usage,
                    )
                    final_answer = rag_answer
            mixed_result: dict[str, object] = {
                "final_answer": final_answer,
                "error": policy_error,
                "standalone_question": latest_user_message or None,
                "citations": self._citations_from_docs(retrieval_docs),
                "reranker_docs": self._serialize_docs(retrieval_docs),
                "context_usage": (
                    {"retrieved_docs_count": len(retrieval_docs)} if retrieval_docs else None
                ),
                "mcp_used": bool(tools_used),
                "mcp_tools_used": tools_used,
                "mcp_tool_invocations": tool_invocations,
            }
            if isinstance(model_id, str) and model_id.strip():
                mixed_result["model_id"] = model_id.strip()
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
                tool_load_started = time.perf_counter()
                mcp_tools = await get_mcp_tools_async(
                    server_keys=mcp_server_keys,
                    run_config=run_cfg,
                )
                logger.info(
                    "chat_runtime_mcp_tools_loaded mode=%s tool_count=%d elapsed_ms=%.1f",
                    normalized_mode,
                    len(mcp_tools),
                    (time.perf_counter() - tool_load_started) * 1000,
                )
                repeated_result = None
                if _repeated_workflow_controller_enabled() and await should_use_repeated_workflow(
                    question=question,
                    tools=mcp_tools,
                    model_id=resolved_model_id,
                    run_config=run_cfg,
                ):
                    repeated_result = await run_repeated_mcp_workflow(
                        question=question,
                        model_id=resolved_model_id,
                        tools=mcp_tools,
                        run_config=run_cfg,
                        require_tool_call=_require_tool_call_enabled(),
                        get_answer=get_mcp_answer_async,
                        checkpoint_path=_workflow_checkpoint_path(),
                        tool_progress_callback=tool_progress_callback,
                        chat_history=chat_history,
                    )
                if repeated_result is None:
                    answer, tools_used, tool_invocations = await get_mcp_answer_async(
                        question,
                        chat_history=chat_history,
                        model_id=resolved_model_id,
                        tools=mcp_tools,
                        run_config=run_cfg,
                        require_tool_call=_require_tool_call_enabled(),
                        tool_progress_callback=tool_progress_callback,
                    )
                else:
                    answer, tools_used, tool_invocations = repeated_result
                mcp_result: dict[str, object] = {
                    "final_answer": answer,
                    "error": None,
                    "standalone_question": question or None,
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": None,
                    "mcp_used": bool(tools_used),
                    "mcp_tools_used": tools_used,
                    "mcp_tool_invocations": tool_invocations,
                }
                tool_failure_error = _tool_failure_summary(
                    cast(list[dict[str, object]], tool_invocations)
                )
                if (
                    _is_trivial_answer(str(mcp_result.get("final_answer") or ""))
                    and tool_failure_error
                ):
                    mcp_result["final_answer"] = tool_failure_error
                    mcp_result["error"] = tool_failure_error
                mcp_result["model_id"] = resolved_model_id
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

                docs = self._retrieve_oracle_docs(
                    query=standalone_question,
                    collection_name=collection_name,
                    k=5,
                )
                docs = self._rank_retrieved_docs(
                    standalone_question,
                    docs,
                    enable_reranker=enable_reranker,
                )
                rag_answer, rag_usage, resolved_model_id = await self._synthesize_rag_answer(
                    question=standalone_question,
                    docs=docs,
                    model_id=model_id,
                    run_config=run_cfg,
                )
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
                    "citations": self._citations_from_docs(docs),
                    "reranker_docs": self._serialize_docs(docs),
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

    async def _stream_runtime_events(
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
    ) -> AsyncIterator[dict[str, object]]:
        result: dict[str, object] = {}
        error: BaseException | None = None
        progress_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        run_done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _emit_tool_progress(payload: dict[str, object]) -> None:
            loop.call_soon_threadsafe(
                lambda: progress_queue.put_nowait({"type": "tool_event", "data": payload}),
            )

        async def _run() -> None:
            nonlocal result, error
            try:
                result = await self.run_chat(
                    messages=messages,
                    model_id=model_id,
                    thread_id=thread_id,
                    session_id=session_id,
                    collection_name=collection_name,
                    enable_reranker=enable_reranker,
                    enable_tracing=enable_tracing,
                    mode=mode,
                    mcp_server_keys=mcp_server_keys,
                    stream=True,
                    tool_progress_callback=_emit_tool_progress,
                )
            except BaseException as exc:  # noqa: BLE001
                error = exc
            finally:
                run_done.set()

        run_task = asyncio.create_task(_run())
        while True:
            if run_done.is_set() and progress_queue.empty():
                break
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                yield progress
            except TimeoutError:
                continue

        await run_task
        if error is not None:
            raise error
        answer = str(result.get("final_answer") or "")
        if answer:
            yield {"type": "text", "delta": answer}

        references: dict[str, object] = {
            "standalone_question": result.get("standalone_question"),
            "citations": result.get("citations") or [],
            "reranker_docs": result.get("reranker_docs") or [],
        }
        if result.get("context_usage") is not None:
            references["context_usage"] = result["context_usage"]
        if result.get("trace_id") is not None:
            references["trace_id"] = result["trace_id"]
        if result.get("mcp_used"):
            references["mcp_used"] = True
        if result.get("mcp_tools_used"):
            references["mcp_tools_used"] = result["mcp_tools_used"]
        invocations = result.get("mcp_tool_invocations")
        if isinstance(invocations, list) and invocations:
            references["mcp_tool_invocations"] = invocations
        if result.get("error"):
            references["error"] = result["error"]
        yield {"type": "references", "data": references}

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

        async for event in self._stream_runtime_events(
            messages=messages,
            model_id=cast(str | None, cfg.get("model_id")),
            thread_id=cast(str | None, cfg.get("thread_id")),
            session_id=cast(str | None, cfg.get("session_id")),
            collection_name=cast(str | None, cfg.get("collection_name")),
            enable_reranker=cast(bool | None, cfg.get("enable_reranker")),
            enable_tracing=cast(bool | None, cfg.get("enable_tracing")),
            mode=cast(str | None, cfg.get("mode")),
            mcp_server_keys=cast(list[str] | None, cfg.get("mcp_server_keys")),
        ):
            event_type = event.get("type")
            if event_type == "text":
                text = str(event.get("delta") or "")
                if not text:
                    continue
                yield v3_raw_event(
                    method="messages",
                    data=(
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": text},
                        },
                        {"langgraph_node": "chat_runtime"},
                    ),
                )
            elif event_type == "tool_event":
                yield v3_raw_event(method="tool_calls", data=event.get("data") or {})
            elif event_type == "references":
                yield v3_raw_event(
                    method="custom",
                    data={"type": "references", "data": event.get("data") or {}},
                )

    def _build_oracle_retrieval_tool(self, collection_name: str | None) -> StructuredTool:
        return rag_runtime.build_oracle_retrieval_tool(
            collection_name=collection_name,
            filter_docs=self._filter_retrieved_docs,
        )

    def _attach_trace_id(
        self,
        result: dict[str, object],
        langfuse_trace: LangfuseChatTrace | None,
    ) -> None:
        if langfuse_trace is not None and langfuse_trace.trace_id:
            result["trace_id"] = langfuse_trace.trace_id

    def _retrieve_oracle_docs(
        self, *, query: str, collection_name: str | None, k: int
    ) -> list[Document]:
        return rag_runtime.retrieve_oracle_docs(query=query, collection_name=collection_name, k=k)

    async def _synthesize_rag_answer(
        self,
        *,
        question: str,
        docs: list[Document],
        model_id: str | None,
        run_config: RunnableConfig | None = None,
    ) -> tuple[str, dict[str, int] | None, str]:
        return await rag_runtime.synthesize_rag_answer(
            question=question,
            docs=docs,
            model_id=model_id,
            run_config=run_config,
        )

    def _filter_retrieved_docs(self, query: str, docs: list[Document]) -> list[Document]:
        return rag_runtime.filter_retrieved_docs(query, docs)

    def _rank_retrieved_docs(
        self,
        query: str,
        docs: list[Document],
        *,
        enable_reranker: bool | None,
    ) -> list[Document]:
        return rag_runtime.rerank_retrieved_docs(
            query,
            docs,
            enable_reranker=enable_reranker,
        )

    def _query_terms(self, query: str) -> list[str]:
        return rag_runtime.query_terms(query)

    def _latest_assistant_answer(
        self, thread_id: str | None, messages: list[dict[str, object]]
    ) -> str | None:
        if thread_id and (thread_state := self._get_thread_state(thread_id)):
            prior_messages = list(thread_state.get("messages") or [])
            for message in reversed(prior_messages):
                content = getattr(message, "content", None)
                if isinstance(message, AIMessage) and isinstance(content, str) and content.strip():
                    return content.strip()

        for item in reversed(messages[:-1]):
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "assistant" and content.strip():
                return content.strip()
        return None

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

    def _serialize_docs(self, docs: list[Document]) -> list[dict[str, object]]:
        return rag_runtime.serialize_docs(docs)

    def _citations_from_docs(self, docs: list[Document]) -> list[dict[str, object]]:
        return rag_runtime.citations_from_docs(docs)

    def _format_retrieved_docs(self, docs: list[Document]) -> str:
        return rag_runtime.format_retrieved_docs(docs)

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
            references: dict[str, object] = {}
            standalone = result.get("standalone_question")
            if isinstance(standalone, str) and standalone.strip():
                references["standalone_question"] = standalone.strip()
            citations = result.get("citations")
            if isinstance(citations, list):
                references["citations"] = citations
            reranker_docs = result.get("reranker_docs")
            if isinstance(reranker_docs, list):
                references["reranker_docs"] = reranker_docs
            context_usage = result.get("context_usage")
            if isinstance(context_usage, dict):
                references["context_usage"] = context_usage
            if result.get("mcp_used") is True:
                references["mcp_used"] = True
            mcp_tools_used = result.get("mcp_tools_used")
            if isinstance(mcp_tools_used, list):
                references["mcp_tools_used"] = [
                    str(tool) for tool in mcp_tools_used if str(tool).strip()
                ]
            mcp_inv = result.get("mcp_tool_invocations")
            if isinstance(mcp_inv, list) and mcp_inv:
                references["mcp_tool_invocations"] = mcp_inv
            error_value = result.get("error")
            if isinstance(error_value, str) and error_value.strip():
                references["error"] = error_value.strip()
            trace_id = result.get("trace_id")
            if isinstance(trace_id, str) and trace_id.strip():
                references["trace_id"] = trace_id.strip()

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

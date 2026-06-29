"""Compatibility chat runtime for direct/RAG execution and stream adaptation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.graphs.mcp_policies import NO_ORACLE_CONTEXT_ANSWER
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.utils.langfuse_tracing import (
    LangfuseChatTrace,
    add_langfuse_callbacks,
    start_langfuse_chat_trace,
)

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

# Configuration helpers.
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
    return "rag"

# Response/reference shaping helpers.
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
class ChatRuntimeService:
    """Compatibility runtime adapter for direct/RAG turns and event streaming."""

    def __init__(
        self,
        thread_state_store: LangGraphCheckpointThreadStateStore | None = None,
    ) -> None:
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
            incoming_messages = messages
            conversation_messages = self._hydrate_thread_messages(thread_id, incoming_messages)
            if normalized_mode == "rag":
                result = await self._run_rag_mode(
                    incoming_messages=incoming_messages,
                    conversation_messages=conversation_messages,
                    model_id=model_id,
                    thread_id=thread_id,
                    session_id=session_id,
                    collection_name=collection_name,
                    enable_reranker=enable_reranker,
                    enable_tracing=enable_tracing,
                    normalized_mode=normalized_mode,
                    mcp_server_keys=mcp_server_keys,
                    stream=stream,
                    answer_delta_callback=answer_delta_callback,
                    langfuse_trace=langfuse_trace,
                )
            elif normalized_mode == "direct":
                result = await self._run_direct_mode(
                    incoming_messages=incoming_messages,
                    conversation_messages=conversation_messages,
                    model_id=model_id,
                    thread_id=thread_id,
                    session_id=session_id,
                    enable_tracing=enable_tracing,
                    normalized_mode=normalized_mode,
                    mcp_server_keys=mcp_server_keys,
                    langfuse_trace=langfuse_trace,
                )
            else:
                raise NotImplementedError(
                    "LangGraph owns mcp and mixed execution. "
                    "Use the chat_agent graph for those modes."
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

    async def _run_direct_mode(
        self,
        *,
        incoming_messages: list[dict[str, object]],
        conversation_messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        enable_tracing: bool | None,
        normalized_mode: str,
        mcp_server_keys: list[str] | None,
        langfuse_trace: LangfuseChatTrace | None = None,
    ) -> dict[str, object]:
        history: list[Any] = []
        latest_question = ""
        for item in conversation_messages:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "user":
                history.append(HumanMessage(content=content))
                latest_question = content.strip() or latest_question
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
            "standalone_question": latest_question or None,
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "model_id": resolved_model_id,
            "usage": emitted_usage,
            "cost_usd": cost_usd,
        }
        if langfuse_trace is not None and langfuse_trace.trace_id:
            direct_result["trace_id"] = langfuse_trace.trace_id
        self._store_thread_state(thread_id, incoming_messages, direct_result)
        return direct_result

    async def _run_rag_mode(
        self,
        *,
        incoming_messages: list[dict[str, object]],
        conversation_messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        collection_name: str | None,
        enable_reranker: bool | None,
        enable_tracing: bool | None,
        normalized_mode: str,
        mcp_server_keys: list[str] | None,
        stream: bool,
        answer_delta_callback: Callable[[str], None] | None = None,
        langfuse_trace: LangfuseChatTrace | None = None,
    ) -> dict[str, object]:
        question = latest_user_message(conversation_messages)
        chat_history = chat_history_before_latest_user(conversation_messages)
        run_cfg = _prepare_run_config(
            thread_id=thread_id,
            mcp_server_keys=mcp_server_keys,
            mode=normalized_mode,
            model_id=model_id,
            session_id=session_id,
            enable_tracing=enable_tracing,
            trace_context=langfuse_trace.trace_context if langfuse_trace else None,
        )
        standalone_question = await contextualize_question(
            question=question,
            chat_history=chat_history,
            model_id=model_id,
            run_config=run_cfg,
        )

        docs = await asyncio.to_thread(
            rag_runtime.retrieve_oracle_docs,
            query=standalone_question,
            collection_name=collection_name,
            k=5,
        )
        docs = await asyncio.to_thread(
            rag_runtime.rerank_retrieved_docs,
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
            rag_answer = NO_ORACLE_CONTEXT_ANSWER
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
        if langfuse_trace is not None and langfuse_trace.trace_id:
            rag_result["trace_id"] = langfuse_trace.trace_id
        self._store_thread_state(thread_id, incoming_messages, rag_result)
        return rag_result

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

    def _hydrate_thread_messages(
        self,
        thread_id: str | None,
        incoming_messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not thread_id:
            return list(incoming_messages)
        return hydrate_thread_messages(self._get_thread_state(thread_id), incoming_messages)

    async def get_state(self, run_config: dict[str, Any]) -> Any:
        thread_id = self._thread_id_from_run_config(run_config)
        values = self._get_thread_state(thread_id) if thread_id else {}
        return type("StateSnapshot", (), {"values": values})()

    def get_state_values(self, run_config: dict[str, Any]) -> dict[str, Any] | None:
        thread_id = self._thread_id_from_run_config(run_config)
        if not thread_id:
            return None
        return self._get_thread_state(thread_id)

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
        incoming_messages = to_langchain_messages(messages)
        updated_messages = prior_messages + new_incoming_messages(
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

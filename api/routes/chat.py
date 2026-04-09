"""RAG chat endpoints: /invoke, /api/chat (stream + non-stream)."""

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any, cast

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.dependencies import (
    build_chat_config,
    generate_request_id,
    get_graph_service,
    log_conversation_in,
    log_conversation_out,
    openai_messages_to_state,
    register_tools_for_run,
    register_tools_for_run_async,
)
from api.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionsRequest,
    ChatCompletionUsage,
    ChatMessage,
    Citation,
    InvokeRequest,
    RerankerDoc,
)
from api.serialization import make_metadata_safe, safe_json
from api.settings import get_settings
from src.rag_agent import State
from src.rag_agent.core import langfuse
from src.rag_agent.core.logging import get_request_id

from ..streaming.ai_sdk_stream import (
    AI_SDK_RESPONSE_HEADERS,
    DONE_FRAME,
    ai_sdk_error_event,
    ai_sdk_sse_frame,
)

logger = logging.getLogger(__name__)


MEDIA_TYPE = "application/json"


def _fresh_turn_state(
    user_request: str,
    chat_history: Sequence[AIMessage | HumanMessage | SystemMessage],
    previous_state: State | None = None,
) -> State:
    history_messages = list(chat_history)
    previous = previous_state or {}
    state = {
        "user_request": user_request,
        "messages": [*history_messages, HumanMessage(content=user_request)],
        "standalone_question": None,
        "history_text": None,
        "mode": None,
        "route": None,
        "retriever_docs": [],
        "reranker_docs": [],
        "citations": [],
        "rag_answer": None,
        "rag_context": None,
        "rag_has_citations": None,
        "mcp_answer": None,
        "direct_answer": None,
        "mcp_used": None,
        "mcp_tools_used": [],
        "mcp_tool_match": None,
        "selected_mcp_tool_names": list(previous.get("selected_mcp_tool_names") or []),
        "selected_mcp_tool_descriptions": list(
            previous.get("selected_mcp_tool_descriptions") or []
        ),
        "context_usage": None,
        "final_answer": None,
        "round": None,
        "max_rounds": None,
        "error": None,
    }
    return cast(State, cast(object, state))


router = APIRouter(tags=["chat"])


def run_rag_and_get_answer(
    messages: list[ChatMessage],
    model_id: str | None = None,
    thread_id: str | None = None,
    session_id: str | None = None,
    collection_name: str | None = None,
    enable_reranker: bool | None = None,
    enable_tracing: bool | None = None,
    mode: str | None = None,
    mcp_server_keys: list[str] | None = None,
    *,
    graph_service: Any,
) -> tuple[
    str,
    str | None,
    str | None,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
    bool,
    list[str],
]:
    user_request, chat_history = openai_messages_to_state(messages)
    log_conversation_in(False, messages, user_request, len(chat_history))
    if not user_request.strip():
        return "", "Empty or missing user message", None, [], [], None, False, []

    run_config = build_chat_config(
        model_id=model_id,
        thread_id=thread_id,
        collection_name=collection_name,
        enable_reranker=enable_reranker,
        enable_tracing=enable_tracing,
        mode=mode,
        mcp_server_keys=mcp_server_keys,
    )
    langfuse.add_langfuse_callbacks(run_config, session_id=session_id, user_id=None)

    previous_values = cast(dict[str, object] | None, graph_service.get_state_values(run_config))
    state = _fresh_turn_state(user_request, chat_history, cast(State | None, previous_values))

    register_tools_for_run(user_request, run_config)
    logger.debug("MCP: sync registration hook invoked before graph invoke")
    t0_rag = time.perf_counter()
    try:
        final_state = graph_service.invoke(state, run_config)  # type: ignore[arg-type]
        logger.info("RAG invoke completed in %.1fs", time.perf_counter() - t0_rag)
        answer = (final_state.get("final_answer") or "").strip()
        err = final_state.get("error")
        standalone = final_state.get("standalone_question") or None
        context_usage = final_state.get("context_usage")
        citations_serializable, docs_serialized, _ = _serialize_state_references(final_state)
        mcp_used = bool(final_state.get("mcp_used"))
        mcp_tools_used = list(final_state.get("mcp_tools_used") or [])
        log_conversation_out(answer or "", err, mcp_used, mcp_tools_used, standalone)
        if err:
            return (
                answer or "",
                str(err),
                standalone,
                citations_serializable,
                docs_serialized,
                context_usage,
                mcp_used,
                mcp_tools_used,
            )
        return (
            answer,
            None,
            standalone,
            citations_serializable,
            docs_serialized,
            context_usage,
            mcp_used,
            mcp_tools_used,
        )
    except Exception as e:
        logger.exception("RAG invoke error: %s", e)
        log_conversation_out("", str(e), None, None, None)
        return "", str(e), None, [], [], None, False, []


def _serialize_state_references(
    vals: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """From graph state values, build serialized citations, reranker_docs, and data_references (AI SDK)."""
    citations = cast(list[dict[str, object]], vals.get("citations") or [])
    reranker_docs = cast(list[dict[str, object]], vals.get("reranker_docs") or [])
    citations_serializable = [
        make_metadata_safe({"source": c.get("source", ""), "page": c.get("page", "")})
        for c in citations
    ]
    docs_serialized = [
        make_metadata_safe(
            {
                "page_content": d.get("page_content", ""),
                "metadata": cast(dict[str, object], (d.get("metadata") or {})),
            }
        )
        for d in reranker_docs
    ]
    data_references: dict[str, object] = {
        "standalone_question": vals.get("standalone_question"),
        "citations": citations_serializable,
        "reranker_docs": docs_serialized,
    }
    if vals.get("context_usage") is not None:
        data_references["context_usage"] = vals["context_usage"]
    if vals.get("mcp_used") is True:
        data_references["mcp_used"] = True
    if vals.get("mcp_tools_used"):
        data_references["mcp_tools_used"] = vals["mcp_tools_used"]
    if vals.get("error"):
        data_references["error"] = vals["error"]
    return citations_serializable, docs_serialized, data_references


def _to_citations(raw: list[dict[str, object]]) -> list[Citation]:
    citations: list[Citation] = []
    for item in raw or []:
        citations.append(
            Citation(
                source=str(item.get("source", "")),
                page=cast(str | None, item.get("page")),
            )
        )
    return citations


def _to_reranker_docs(raw: list[dict[str, object]]) -> list[RerankerDoc]:
    docs: list[RerankerDoc] = []
    for doc in raw or []:
        docs.append(
            RerankerDoc(
                page_content=str(doc.get("page_content", "")),
                metadata=cast(dict[str, Any], (doc.get("metadata") or {})),
            )
        )
    return docs


def chat_completion_response_json(
    content: str,
    model_id: str | None,
    completion_id: str,
    standalone_question: str | None = None,
    citations: list[dict[str, object]] | None = None,
    reranker_docs: list[dict[str, object]] | None = None,
    context_usage: dict[str, object] | None = None,
) -> dict[str, object]:
    response = ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=model_id or get_settings().LLM_MODEL_ID,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": content},
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
        content=content,
        standalone_question=standalone_question,
        citations=_to_citations(citations or []) if citations is not None else None,
        reranker_docs=_to_reranker_docs(reranker_docs or []) if reranker_docs is not None else None,
        context_usage=context_usage,
    )
    return response.model_dump()


async def stream_rag_ai_sdk_sse(
    messages: list[ChatMessage],
    model_id: str | None,
    thread_id: str | None,
    session_id: str | None = None,
    collection_name: str | None = None,
    enable_reranker: bool | None = None,
    enable_tracing: bool | None = None,
    mode: str | None = None,
    mcp_server_keys: list[str] | None = None,
    *,
    graph_service: Any,
):
    """Emit AI SDK UI message stream parts while running the RAG graph (no legacy metadata)."""
    user_request, chat_history = openai_messages_to_state(messages)
    log_conversation_in(True, messages, user_request, len(chat_history))
    if not user_request.strip():
        yield ai_sdk_sse_frame({"type": "error", "errorText": "Empty or missing user message"})
        yield DONE_FRAME
        return
    run_config = build_chat_config(
        model_id=model_id,
        thread_id=thread_id,
        collection_name=collection_name,
        enable_reranker=enable_reranker,
        enable_tracing=enable_tracing,
        mode=mode,
        mcp_server_keys=mcp_server_keys,
    )
    langfuse.add_langfuse_callbacks(run_config, session_id=session_id, user_id=None)
    await register_tools_for_run_async(user_request, run_config)
    logger.debug("MCP: async registration hook invoked before graph stream")

    message_id = uuid.uuid4().hex
    text_id = f"{message_id}:text"
    yield ai_sdk_sse_frame({"type": "start", "messageId": message_id})
    yield ai_sdk_sse_frame({"type": "text-start", "id": text_id})

    state = _fresh_turn_state(user_request, chat_history)

    try:
        async for event in graph_service.astream(
            state,
            run_config,  # type: ignore[arg-type]
        ):
            if not (isinstance(event, tuple) and len(event) == 2):
                continue
            mode_name, chunk = event
            if mode_name == "updates":
                for node_name, update in chunk.items():
                    if node_name == "SearchErrorResponse":
                        text = update.get("final_answer") or ""
                        if text:
                            yield ai_sdk_sse_frame(
                                {"type": "text-delta", "id": text_id, "delta": text}
                            )
                    elif node_name == "DraftAnswer":
                        text = update.get("final_answer") or ""
                        if text:
                            chunk_size = 32
                            for i in range(0, len(text), chunk_size):
                                yield ai_sdk_sse_frame(
                                    {
                                        "type": "text-delta",
                                        "id": text_id,
                                        "delta": text[i : i + chunk_size],
                                    }
                                )
            elif mode_name == "messages":
                msg, metadata = chunk
                node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                if node in (
                    "DraftAnswer",
                    "Rerank",
                    "AnswerFromDocs",
                    "CallMCPTools",
                    "DirectAnswer",
                    "FollowUpInterpreter",
                    "GroundedReformatAnswer",
                    "Router",
                    "Moderator",
                    "Search",
                    "SelectMCPTools",
                ):
                    continue
                if node and getattr(msg, "content", None):
                    yield ai_sdk_sse_frame(
                        {"type": "text-delta", "id": text_id, "delta": msg.content}
                    )
        yield ai_sdk_sse_frame({"type": "text-end", "id": text_id})
        try:
            state_snapshot = await graph_service.get_state(run_config)  # type: ignore[arg-type]
            vals = getattr(state_snapshot, "values", None) or {}
            standalone = vals.get("standalone_question") or None
            final_answer = (vals.get("final_answer") or "").strip()
            state_error = vals.get("error")
            log_conversation_out(
                final_answer,
                state_error,
                vals.get("mcp_used"),
                vals.get("mcp_tools_used"),
                standalone,
            )
            citations_serializable, docs_serialized, data_references = _serialize_state_references(
                vals
            )
            context_usage = vals.get("context_usage")
            mcp_used = vals.get("mcp_used")
            mcp_tools_used = vals.get("mcp_tools_used") or []

            if (
                standalone is not None
                or citations_serializable
                or docs_serialized
                or context_usage is not None
                or mcp_used is True
                or mcp_tools_used
                or state_error
            ):
                yield ai_sdk_sse_frame(
                    {
                        "type": "data-references",
                        "id": f"ref-{message_id}",
                        "data": data_references,
                        "transient": False,
                    }
                )
        except Exception as e:
            logger.warning("Could not get final state for data-references: %s", e)
        yield ai_sdk_sse_frame({"type": "finish", "finishReason": "stop"})
        langfuse.safe_flush()
        yield DONE_FRAME
    except Exception as e:
        logger.exception("RAG AI SDK stream error: %s", e)
        yield ai_sdk_sse_frame(ai_sdk_error_event("Internal server error"))
        yield DONE_FRAME


async def stream_graph_updates(
    user_input: str, run_config: dict[str, object] | None = None, *, graph_service: Any
):
    state = State(user_request=user_input, messages=[])
    c = run_config or build_chat_config()
    async for step_output in graph_service.astream(state, c):  # type: ignore[arg-type]
        yield safe_json(step_output) + "\n"


@router.post("/invoke")
async def invoke(request: InvokeRequest, graph_service: Any = Depends(get_graph_service)):
    _config = build_chat_config()
    if get_settings().DEBUG:
        safe = {
            k: v
            for k, v in (_config.get("configurable") or {}).items()
            if k not in ("mcp_url", "mcp_server_keys")
        }
        logger.debug("Invoked Agent API config (sanitized): %s", safe)
    try:
        return StreamingResponse(
            stream_graph_updates(request.user_input, _config, graph_service=graph_service),
            media_type=MEDIA_TYPE,
        )
    except Exception as e:
        logger.exception("Error in invoke endpoint: %s", e)
        return {"error": str(e)}


@router.post("/api/chat")
async def chat_completions(
    request: ChatCompletionsRequest, graph_service: Any = Depends(get_graph_service)
):
    """Chat: RAG + optional MCP (tools in LangChain). Single endpoint at /api/chat."""
    effective_reranker = (
        request.enable_reranker
        if request.enable_reranker is not None
        else getattr(get_settings(), "ENABLE_RERANKER", True)
    )
    logger.info(
        "Chat request: enable_reranker=%s (from UI: %s)",
        effective_reranker,
        request.enable_reranker,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    effective_thread_id = request.thread_id or generate_request_id()

    if request.stream:
        return StreamingResponse(
            stream_rag_ai_sdk_sse(
                request.messages,
                request.model,
                effective_thread_id,
                request.session_id,
                request.collection_name,
                request.enable_reranker,
                request.enable_tracing,
                request.mode,
                request.mcp_server_keys,
                graph_service=graph_service,
            ),
            headers=AI_SDK_RESPONSE_HEADERS,
        )

    # Ensure request_id ContextVar propagates into thread used by asyncio.to_thread.
    # While Python 3.11's to_thread typically copies contextvars, we defensively bind
    # the current request_id inside the worker thread to guarantee correlation.
    def _run_with_request_id(req_id: str):
        from src.rag_agent.core.logging import REQUEST_ID_CTX

        token = REQUEST_ID_CTX.set(req_id)
        try:
            return run_rag_and_get_answer(
                request.messages,
                request.model,
                effective_thread_id,
                request.session_id,
                request.collection_name,
                request.enable_reranker,
                request.enable_tracing,
                request.mode,
                request.mcp_server_keys,
                graph_service=graph_service,
            )
        finally:
            REQUEST_ID_CTX.reset(token)

    current_req_id = get_request_id()
    (
        answer,
        err,
        standalone,
        citations,
        reranker_docs,
        context_usage,
        _mcp_used,
        _mcp_tools_used,
    ) = await asyncio.to_thread(_run_with_request_id, current_req_id)
    if err:
        langfuse.safe_flush()
        return JSONResponse(
            content={"error": err, "content": answer or "", "thread_id": effective_thread_id},
            status_code=503,
        )
    resp = chat_completion_response_json(
        answer,
        request.model,
        completion_id,
        standalone_question=standalone,
        citations=citations,
        reranker_docs=reranker_docs,
        context_usage=context_usage,
    )
    if isinstance(resp, dict):
        resp["thread_id"] = effective_thread_id
    langfuse.safe_flush()
    return resp


@router.delete("/api/threads/{thread_id}")
async def delete_thread(
    thread_id: str, graph_service: Any = Depends(get_graph_service)
) -> Response:
    run_config = build_chat_config(thread_id=thread_id)
    state_snapshot = await graph_service.get_state(run_config)
    if not state_snapshot.values:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    await graph_service.delete_thread(thread_id)
    return Response(status_code=204)

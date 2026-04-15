"""LangGraph-style thread/run router served by the FastAPI app.

This module provides the API surface used by frontend ``useStream`` clients
and delegates runtime execution to ``RuntimeAgent``.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from fastapi import APIRouter, Depends, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, model_validator

from api.dependencies import generate_request_id, get_graph_service
from api.serialization import make_metadata_safe
from src.rag_agent.core.citations import normalize_citations
from src.rag_agent.runtime.agent import RuntimeAgent
from src.rag_agent.runtime.middleware import merge_runtime_context
from src.rag_agent.runtime.responses import chat_completion_response_json

router = APIRouter(tags=["langgraph-runtime"])


class ThreadCreateRequest(BaseModel):
    thread_id: str | None = None


class ThreadCreateResponse(BaseModel):
    thread_id: str


class RunInput(BaseModel):
    messages: list[dict[str, Any]] | None = None
    message: str | None = None
    model: str | None = None
    session_id: str | None = None
    collection_name: str | None = None
    enable_reranker: bool | None = None
    enable_tracing: bool | None = None
    mode: str | None = None
    mcp_server_keys: list[str] | None = None

    @staticmethod
    def _normalized_role(message: dict[str, Any]) -> str:
        role = message.get("role")
        if isinstance(role, str):
            normalized = role.strip().lower()
            if normalized:
                return normalized
        msg_type = message.get("type")
        if isinstance(msg_type, str):
            lowered = msg_type.strip().lower()
            if lowered == "human":
                return "user"
            if lowered == "ai":
                return "assistant"
            if lowered == "system":
                return "system"
        return ""

    @model_validator(mode="after")
    def _validate_user_input(self) -> RunInput:
        if self.messages and len(self.messages) > 0:
            roles = [self._normalized_role(message) for message in self.messages]
            if "user" not in roles:
                raise ValueError("messages must include at least one user/human message.")
            if roles[-1] != "user":
                raise ValueError("last message must be user/human.")
            return self
        if isinstance(self.message, str) and self.message.strip():
            return self
        raise ValueError("Provide either non-empty messages or message.")


class ThreadRunRequest(BaseModel):
    input: RunInput | None = None
    messages: list[dict[str, Any]] | None = None
    message: str | None = None
    model: str | None = None
    session_id: str | None = None
    collection_name: str | None = None
    enable_reranker: bool | None = None
    enable_tracing: bool | None = None
    mode: str | None = None
    mcp_server_keys: list[str] | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    configurable: dict[str, Any] | None = None
    assistant_id: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> ThreadRunRequest:
        if self.input is not None:
            return self
        if self.messages and len(self.messages) > 0:
            return self
        if isinstance(self.message, str) and self.message.strip():
            return self
        raise ValueError("Provide input, messages, or message.")


class ThreadRunResponse(BaseModel):
    run_id: str
    thread_id: str
    output: dict[str, Any]


class ThreadHistoryRequest(BaseModel):
    limit: int | None = None
    before: str | None = None
    metadata: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None


def _to_stream_message(
    *,
    role: str,
    content: str,
    message_id: str | None = None,
    references: dict[str, object] | None = None,
) -> dict[str, object]:
    stream_type = "human" if role == "user" else "ai" if role == "assistant" else "system"
    message: dict[str, object] = {"type": stream_type, "content": content}
    if message_id:
        message["id"] = message_id
    if references:
        message["additional_kwargs"] = references
        message["response_metadata"] = references
    return message


def _serialize_state_messages(raw_messages: object) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []
    serialized: list[dict[str, Any]] = []
    for item in raw_messages:
        role: str | None = None
        content: str | None = None
        additional_kwargs: dict[str, Any] | None = None
        response_metadata: dict[str, Any] | None = None
        if isinstance(item, dict):
            item_role = item.get("role")
            item_content = item.get("content")
            if isinstance(item_role, str):
                lowered = item_role.strip().lower()
                if lowered in {"user", "assistant", "system"}:
                    role = lowered
            if isinstance(item_content, str):
                content = item_content
            raw_additional = item.get("additional_kwargs")
            raw_metadata = item.get("response_metadata")
            if isinstance(raw_additional, dict):
                additional_kwargs = cast(dict[str, Any], raw_additional)
            if isinstance(raw_metadata, dict):
                response_metadata = cast(dict[str, Any], raw_metadata)
        else:
            msg_type = str(getattr(item, "type", "") or "").strip().lower()
            if msg_type == "human":
                role = "user"
            elif msg_type == "ai":
                role = "assistant"
            elif msg_type == "system":
                role = "system"
            raw_content = getattr(item, "content", None)
            if isinstance(raw_content, str):
                content = raw_content
            raw_additional = getattr(item, "additional_kwargs", None)
            raw_metadata = getattr(item, "response_metadata", None)
            if isinstance(raw_additional, dict):
                additional_kwargs = cast(dict[str, Any], raw_additional)
            if isinstance(raw_metadata, dict):
                response_metadata = cast(dict[str, Any], raw_metadata)
        if role and content is not None:
            message: dict[str, Any] = {"role": role, "content": content}
            if additional_kwargs:
                message["additional_kwargs"] = additional_kwargs
            if response_metadata:
                message["response_metadata"] = response_metadata
            serialized.append(message)
    return serialized


def _effective_run_input(payload: ThreadRunRequest) -> RunInput:
    if payload.input is not None:
        return payload.input

    merged = merge_runtime_context(
        top_level={
            "messages": payload.messages,
            "message": payload.message,
            "model": payload.model,
            "session_id": payload.session_id,
            "collection_name": payload.collection_name,
            "enable_reranker": payload.enable_reranker,
            "enable_tracing": payload.enable_tracing,
            "mode": payload.mode,
            "mcp_server_keys": payload.mcp_server_keys,
        },
        context=payload.context,
        metadata=payload.metadata,
        configurable=payload.configurable,
    )
    return RunInput(**merged)


@router.post("/api/langgraph/threads", response_model=ThreadCreateResponse)
async def create_thread(payload: ThreadCreateRequest) -> ThreadCreateResponse:
    thread_id = payload.thread_id or generate_request_id()
    return ThreadCreateResponse(thread_id=thread_id)


@router.post("/api/langgraph/threads/{thread_id}/runs/stream")
async def stream_thread_run(
    thread_id: str,
    request: ThreadRunRequest,
    chat_runtime_service: Any = Depends(get_graph_service),
) -> StreamingResponse:
    run_input = _effective_run_input(request)
    _ = request.assistant_id
    runtime_agent = RuntimeAgent(chat_runtime_service)
    messages = runtime_agent.normalize_messages(run_input.messages, run_input.message)

    async def _stream() -> Any:
        turn_id = uuid.uuid4().hex[:12]
        assistant_message_id = f"{thread_id}:assistant:{turn_id}"
        assistant_text = ""
        references: dict[str, object] = {}
        base_messages: list[dict[str, object]] = []

        try:
            state_snapshot = await chat_runtime_service.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
            values = cast(dict[str, Any], getattr(state_snapshot, "values", None) or {})
            historical = _serialize_state_messages(values.get("messages"))
            for idx, message in enumerate(historical):
                role = str(message.get("role") or "").strip().lower()
                content = str(message.get("content") or "")
                if not role or not content:
                    continue
                base_messages.append(
                    _to_stream_message(
                        role=role,
                        content=content,
                        message_id=f"{thread_id}:hist:{idx}",
                    )
                )
        except Exception:
            base_messages = []

        for idx, message in enumerate(messages):
            role = str(message.role or "").strip().lower()
            content = str(message.content or "")
            if not role or not content:
                continue
            pending = _to_stream_message(
                role=role,
                content=content,
                message_id=f"{thread_id}:pending:{turn_id}:{idx}",
            )
            last = base_messages[-1] if base_messages else None
            if last and last.get("type") == pending.get("type") and last.get("content") == pending.get(
                "content"
            ):
                continue
            base_messages.append(pending)

        if base_messages:
            yield f"event: values\ndata: {JSONResponse(content={'messages': base_messages}).body.decode()}\n\n"

        def _emit_values() -> str:
            payload_messages = list(base_messages)
            if assistant_text or references:
                payload_messages.append(
                    _to_stream_message(
                        role="assistant",
                        content=assistant_text,
                        message_id=assistant_message_id,
                        references=references,
                    )
                )
            payload = {"messages": payload_messages}
            return f"event: values\ndata: {JSONResponse(content=payload).body.decode()}\n\n"

        try:
            async for event in runtime_agent.stream(
                messages=messages,
                model_id=run_input.model,
                thread_id=thread_id,
                session_id=run_input.session_id,
                collection_name=run_input.collection_name,
                enable_reranker=run_input.enable_reranker,
                enable_tracing=run_input.enable_tracing,
                mode=run_input.mode,
                mcp_server_keys=run_input.mcp_server_keys,
            ):
                if event.get("type") == "text":
                    delta = str(event.get("delta") or "")
                    if delta:
                        assistant_text += delta
                        yield _emit_values()
                elif event.get("type") == "references":
                    safe_references = make_metadata_safe(
                        cast(dict[str, object], event.get("data") or {})
                    )
                    citations = normalize_citations(
                        cast(list[dict[str, object]], safe_references.get("citations") or [])
                    )
                    safe_references["citations"] = citations
                    references = cast(dict[str, object], safe_references)
                    yield _emit_values()
        except Exception:
            error_references = dict(references)
            error_references["error"] = "Internal server error"
            references = error_references
            yield _emit_values()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
            "connection": "keep-alive",
        },
    )


@router.get("/api/langgraph/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    chat_runtime_service: Any = Depends(get_graph_service),
) -> JSONResponse:
    state_snapshot = await chat_runtime_service.get_state(
        {"configurable": {"thread_id": thread_id}}
    )
    values = cast(dict[str, Any], getattr(state_snapshot, "values", None) or {})
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            {
            "values": {
                "messages": _serialize_state_messages(values.get("messages")),
            },
            "next": [],
            "tasks": [],
            "checkpoint": None,
            "metadata": {},
            "created_at": None,
            "parent_checkpoint": None,
            }
        ),
    )


@router.post("/api/langgraph/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    _: ThreadHistoryRequest,
    chat_runtime_service: Any = Depends(get_graph_service),
) -> JSONResponse:
    state_snapshot = await chat_runtime_service.get_state(
        {"configurable": {"thread_id": thread_id}}
    )
    values = cast(dict[str, Any], getattr(state_snapshot, "values", None) or {})
    messages = _serialize_state_messages(values.get("messages"))
    if not messages:
        return JSONResponse(status_code=200, content=[])

    # Minimal LangGraph-compatible history payload: newest state first.
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            [
                {
                    "values": {"messages": messages},
                    "next": [],
                    "tasks": [],
                    "checkpoint": None,
                    "metadata": {},
                    "created_at": None,
                    "parent_checkpoint": None,
                }
            ]
        ),
    )


@router.post("/api/langgraph/threads/{thread_id}/runs", response_model=ThreadRunResponse)
async def run_thread(
    thread_id: str,
    request: ThreadRunRequest,
    chat_runtime_service: Any = Depends(get_graph_service),
) -> ThreadRunResponse | JSONResponse:
    run_input = _effective_run_input(request)
    _ = request.assistant_id
    runtime_agent = RuntimeAgent(chat_runtime_service)
    messages = runtime_agent.normalize_messages(run_input.messages, run_input.message)

    result = await runtime_agent.invoke(
        messages=messages,
        model_id=run_input.model,
        thread_id=thread_id,
        session_id=run_input.session_id,
        collection_name=run_input.collection_name,
        enable_reranker=run_input.enable_reranker,
        enable_tracing=run_input.enable_tracing,
        mode=run_input.mode,
        mcp_server_keys=run_input.mcp_server_keys,
    )
    answer = str(result.get("final_answer") or "").strip()
    err = cast(str | None, result.get("error"))
    standalone = cast(str | None, result.get("standalone_question"))
    citations = cast(list[dict[str, object]], result.get("citations") or [])
    reranker_docs = cast(list[dict[str, object]], result.get("reranker_docs") or [])
    context_usage = cast(dict[str, object] | None, result.get("context_usage"))
    usage = cast(dict[str, object] | None, result.get("usage"))
    resolved_model_id = cast(str | None, result.get("model_id")) or run_input.model

    if err:
        return JSONResponse(
            status_code=503,
            content={
                "run_id": f"run-{uuid.uuid4().hex[:24]}",
                "thread_id": thread_id,
                "output": {"error": err, "content": answer or ""},
            },
        )
    output = chat_completion_response_json(
        content=answer,
        model_id=resolved_model_id,
        completion_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        standalone_question=standalone,
        citations=citations,
        reranker_docs=reranker_docs,
        context_usage=context_usage,
        usage=usage,
    )
    return ThreadRunResponse(
        run_id=f"run-{uuid.uuid4().hex[:24]}",
        thread_id=thread_id,
        output=cast(dict[str, Any], output),
    )


@router.delete("/api/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str, chat_runtime_service: Any = Depends(get_graph_service)
) -> Response:
    await chat_runtime_service.delete_thread(thread_id)
    return Response(status_code=204)

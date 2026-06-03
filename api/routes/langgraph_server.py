"""LangGraph-style thread/run router served by the FastAPI app.

This module provides the API surface used by frontend ``useStream`` clients
and delegates runtime execution to ``ChatRuntimeService``.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field, model_validator

from api.dependencies import generate_request_id, log_conversation_out
from api.deps.request import get_graph_service
from api.routes.langgraph_middleware import merge_runtime_context
from api.schemas import ChatMessage
from api.serialization import make_metadata_safe
from src.rag_agent.core.citations import normalize_citations
from src.rag_agent.runtime.agent import normalize_messages

router = APIRouter(tags=["langgraph-runtime"])
logger = logging.getLogger(__name__)


def _json_response_body(content: object) -> str:
    body = JSONResponse(content=content).body
    if isinstance(body, memoryview):
        body = body.tobytes()
    return body.decode()


def _stream_input_payload(messages: list[ChatMessage]) -> dict[str, object]:
    return {"messages": [message.model_dump() for message in messages]}


def _stream_config(
    *,
    thread_id: str,
    run_input: RunInput,
) -> dict[str, object]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "model_id": run_input.model,
            "session_id": run_input.session_id,
            "collection_name": run_input.collection_name,
            "enable_reranker": run_input.enable_reranker,
            "enable_tracing": run_input.enable_tracing,
            "mode": run_input.mode,
            "mcp_server_keys": run_input.mcp_server_keys,
        }
    }


async def _astream_v3_raw_events(
    runtime_service: Any,
    *,
    messages: list[ChatMessage],
    thread_id: str,
    run_input: RunInput,
) -> AsyncIterator[dict[str, object]]:
    event_streamer = getattr(runtime_service, "astream_events", None)
    if not callable(event_streamer):
        raise AttributeError("Runtime service must expose astream_events(..., version='v3').")

    raw_stream = event_streamer(
        _stream_input_payload(messages),
        config=_stream_config(thread_id=thread_id, run_input=run_input),
        version="v3",
    )
    if inspect.isawaitable(raw_stream):
        raw_stream = await raw_stream

    async for raw_event in raw_stream:
        yield cast(dict[str, object], raw_event)


def _events_from_v3(raw_event: object) -> list[dict[str, object]]:
    if not isinstance(raw_event, dict):
        return []
    method = raw_event.get("method")
    params = raw_event.get("params")
    if not isinstance(params, dict):
        return []
    data = params.get("data")
    if method == "messages":
        return _events_from_v3_message(data)
    if method == "tool_calls":
        safe_data = data if isinstance(data, dict) else {"event": str(data)}
        return [{"type": "tool_event", "data": cast(dict[str, object], safe_data)}]
    if method == "custom":
        return _events_from_v3_custom(data)
    return []


def _events_from_v3_message(data: object) -> list[dict[str, object]]:
    payload: object
    if isinstance(data, tuple) and data:
        payload = data[0]
    else:
        payload = data
    if not isinstance(payload, dict):
        return []

    event_name = payload.get("event")
    if event_name != "content-block-delta":
        return []
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return []
    delta_type = delta.get("type")
    if delta_type == "text-delta":
        text = delta.get("text")
        if isinstance(text, str) and text:
            return [{"type": "text", "delta": text}]
    if delta_type in {"tool-call-delta", "tool_call_chunk"}:
        return [{"type": "tool_event", "data": cast(dict[str, object], delta)}]
    return []


def _events_from_v3_custom(data: object) -> list[dict[str, object]]:
    if not isinstance(data, dict):
        return []
    event_type = data.get("type")
    if event_type not in {"references", "tool_event"}:
        return []
    payload = data.get("data")
    if not isinstance(payload, dict):
        payload = {}
    return [{"type": cast(Literal["references", "tool_event"], event_type), "data": payload}]


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
    stream_mode: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("stream_mode", "streamMode"),
    )

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
    stream_mode: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("stream_mode", "streamMode"),
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> ThreadRunRequest:
        if self.input is not None:
            return self
        if self.messages and len(self.messages) > 0:
            return self
        if isinstance(self.message, str) and self.message.strip():
            return self
        raise ValueError("Provide input, messages, or message.")


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


def _stream_error_message() -> str:
    return "Internal server error"


def _safe_non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _to_tools_stream_event(data: dict[str, object]) -> dict[str, object] | None:
    """Convert internal tool progress metadata to LangGraph SDK tools events."""

    raw_event = _safe_non_empty_string(data.get("event"))
    if raw_event in {"on_tool_start", "on_tool_event", "on_tool_end", "on_tool_error"}:
        name = _safe_non_empty_string(data.get("name"))
        if not name:
            return None
        event_payload: dict[str, object] = {"event": raw_event, "name": name}
        tool_call_id = _safe_non_empty_string(data.get("toolCallId"))
        if tool_call_id:
            event_payload["toolCallId"] = tool_call_id
        for key in ("input", "data", "output", "error"):
            if key in data:
                event_payload[key] = data[key]
        return event_payload

    phase = _safe_non_empty_string(data.get("phase"))
    tool_name = _safe_non_empty_string(data.get("tool_name"))
    if not phase or not tool_name:
        return None

    phase_event_payload: dict[str, object] = {"name": tool_name}
    tool_run_id = _safe_non_empty_string(data.get("tool_run_id"))
    if tool_run_id:
        phase_event_payload["toolCallId"] = tool_run_id

    if phase == "start":
        phase_event_payload["event"] = "on_tool_start"
        phase_event_payload["input"] = data.get("args", {})
        return phase_event_payload
    if phase == "end":
        phase_event_payload["event"] = "on_tool_end"
        phase_event_payload["output"] = data.get("result")
        return phase_event_payload
    if phase == "error":
        phase_event_payload["event"] = "on_tool_error"
        phase_event_payload["error"] = data.get("error") or data.get("result")
        return phase_event_payload
    return None


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
            "stream_mode": payload.stream_mode,
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
    messages = normalize_messages(run_input.messages, run_input.message)

    async def _stream() -> Any:
        turn_id = uuid.uuid4().hex[:12]
        assistant_message_id = f"{thread_id}:assistant:{turn_id}"
        assistant_text = ""
        references: dict[str, object] = {}
        progress_events: list[dict[str, object]] = []
        base_messages: list[dict[str, object]] = []

        try:
            state_snapshot = await chat_runtime_service.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
            values = cast(dict[str, Any], getattr(state_snapshot, "values", None) or {})
            historical = _serialize_state_messages(values.get("messages"))
            for idx, history_message in enumerate(historical):
                role = str(history_message.get("role") or "").strip().lower()
                content = str(history_message.get("content") or "")
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

        for idx, pending_message in enumerate(messages):
            role = str(pending_message.role or "").strip().lower()
            content = str(pending_message.content or "")
            if not role or not content:
                continue
            pending = _to_stream_message(
                role=role,
                content=content,
                message_id=f"{thread_id}:pending:{turn_id}:{idx}",
            )
            last = base_messages[-1] if base_messages else None
            if (
                last
                and last.get("type") == pending.get("type")
                and last.get("content") == pending.get("content")
            ):
                continue
            base_messages.append(pending)

        if base_messages:
            yield f"event: values\ndata: {_json_response_body({'messages': base_messages})}\n\n"

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
            return f"event: values\ndata: {_json_response_body(payload)}\n\n"

        try:
            async for raw_event in _astream_v3_raw_events(
                chat_runtime_service,
                messages=messages,
                thread_id=thread_id,
                run_input=run_input,
            ):
                for event in _events_from_v3(raw_event):
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
                        if progress_events:
                            safe_references["mcp_progress_events"] = list(progress_events)
                        references = cast(dict[str, object], safe_references)
                        yield _emit_values()
                    elif event.get("type") == "tool_event":
                        safe_event = make_metadata_safe(
                            cast(dict[str, object], event.get("data") or {})
                        )
                        progress_events.append(cast(dict[str, object], safe_event))
                        if len(progress_events) > 100:
                            progress_events = progress_events[-100:]
                        references["mcp_progress_events"] = list(progress_events)
                        tools_stream_event = _to_tools_stream_event(
                            cast(dict[str, object], safe_event)
                        )
                        if tools_stream_event:
                            yield (
                                "event: tools\n"
                                f"data: {_json_response_body(tools_stream_event)}\n\n"
                            )
                        yield _emit_values()
            log_conversation_out(
                final_answer=assistant_text,
                error=cast(str | None, references.get("error")),
                mcp_used=cast(bool | None, references.get("mcp_used")),
                mcp_tools_used=cast(
                    list[Any] | None,
                    references.get("mcp_tools_used"),
                ),
                standalone_question=cast(str | None, references.get("standalone_question")),
            )
        except Exception:
            logger.exception("langgraph_stream_run_failed thread_id=%s", thread_id)
            error_references = dict(references)
            error_references["error"] = _stream_error_message()
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


@router.delete("/api/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str, chat_runtime_service: Any = Depends(get_graph_service)
) -> Response:
    await chat_runtime_service.delete_thread(thread_id)
    return Response(status_code=204)

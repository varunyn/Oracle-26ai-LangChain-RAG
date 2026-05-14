"""Helpers for LangChain/LangGraph v3 runtime event streams."""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from api.schemas import ChatMessage


def v3_raw_event(*, method: str, data: object) -> dict[str, object]:
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": [],
            "timestamp": int(time.time() * 1000),
            "data": data,
        },
    }


def stream_input_payload(messages: list[ChatMessage]) -> dict[str, object]:
    return {"messages": [message.model_dump() for message in messages]}


def stream_config(
    *,
    thread_id: str,
    model_id: str | None,
    session_id: str | None,
    collection_name: str | None,
    enable_reranker: bool | None,
    enable_tracing: bool | None,
    mode: str | None,
    mcp_server_keys: list[str] | None,
) -> dict[str, object]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "model_id": model_id,
            "session_id": session_id,
            "collection_name": collection_name,
            "enable_reranker": enable_reranker,
            "enable_tracing": enable_tracing,
            "mode": mode,
            "mcp_server_keys": mcp_server_keys,
        }
    }


async def astream_v3_raw_events(
    runtime_service: Any,
    *,
    messages: list[ChatMessage],
    model_id: str | None,
    thread_id: str,
    session_id: str | None,
    collection_name: str | None,
    enable_reranker: bool | None,
    enable_tracing: bool | None,
    mode: str | None,
    mcp_server_keys: list[str] | None,
) -> AsyncIterator[dict[str, object]]:
    event_streamer = getattr(runtime_service, "astream_events", None)
    if not callable(event_streamer):
        raise AttributeError("Runtime service must expose astream_events(..., version='v3').")

    raw_stream = event_streamer(
        stream_input_payload(messages),
        config=stream_config(
            thread_id=thread_id,
            model_id=model_id,
            session_id=session_id,
            collection_name=collection_name,
            enable_reranker=enable_reranker,
            enable_tracing=enable_tracing,
            mode=mode,
            mcp_server_keys=mcp_server_keys,
        ),
        version="v3",
    )
    if inspect.isawaitable(raw_stream):
        raw_stream = await raw_stream

    async for raw_event in raw_stream:
        yield cast(dict[str, object], raw_event)


def runtime_events_from_v3(raw_event: object) -> list[dict[str, object]]:
    if not isinstance(raw_event, dict):
        return []
    method = raw_event.get("method")
    params = raw_event.get("params")
    if not isinstance(params, dict):
        return []
    data = params.get("data")
    if method == "messages":
        return _runtime_events_from_v3_message(data)
    if method == "tool_calls":
        safe_data = data if isinstance(data, dict) else {"event": str(data)}
        return [{"type": "tool_event", "data": cast(dict[str, object], safe_data)}]
    if method == "custom":
        return _runtime_events_from_v3_custom(data)
    return []


def _runtime_events_from_v3_message(data: object) -> list[dict[str, object]]:
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


def _runtime_events_from_v3_custom(data: object) -> list[dict[str, object]]:
    if not isinstance(data, dict):
        return []
    event_type = data.get("type")
    if event_type not in {"references", "tool_event"}:
        return []
    payload = data.get("data")
    if not isinstance(payload, dict):
        payload = {}
    return [{"type": cast(Literal["references", "tool_event"], event_type), "data": payload}]

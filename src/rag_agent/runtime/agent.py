"""Runtime agent facade for thread/run APIs.

This module centralizes request-shape normalization and delegates execution to
``ChatRuntimeService``. It is intentionally framework-agnostic so route modules
can reuse the same invocation and streaming behavior.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from api.schemas import ChatMessage


class RuntimeAgent:
    """Facade over ChatRuntimeService for invoke and streaming run APIs."""

    def __init__(self, chat_runtime_service: Any) -> None:
        self._chat_runtime_service = chat_runtime_service

    @staticmethod
    def normalize_messages(messages: list[dict[str, Any]] | None, message: str | None) -> list[ChatMessage]:
        if messages:
            normalized: list[ChatMessage] = []
            for item in messages:
                role_raw = item.get("role")
                if not isinstance(role_raw, str):
                    msg_type = item.get("type")
                    if msg_type == "human":
                        role_raw = "user"
                    elif msg_type == "ai":
                        role_raw = "assistant"
                    elif msg_type == "system":
                        role_raw = "system"
                content_raw = item.get("content")
                if isinstance(content_raw, list):
                    content_raw = "".join(
                        block.get("text", "")
                        for block in content_raw
                        if isinstance(block, dict) and isinstance(block.get("text"), str)
                    )
                if role_raw in {"user", "assistant", "system"} and isinstance(content_raw, str):
                    role = cast(Literal["user", "assistant", "system"], role_raw)
                    normalized.append(ChatMessage(role=role, content=content_raw))
            if normalized:
                return normalized
        return [ChatMessage(role="user", content=str(message or "").strip())]

    async def invoke(
        self,
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
    ) -> dict[str, object]:
        return cast(
            dict[str, object],
            await self._chat_runtime_service.run_chat(
                messages=[message.model_dump() for message in messages],
                model_id=model_id,
                thread_id=thread_id,
                session_id=session_id,
                collection_name=collection_name,
                enable_reranker=enable_reranker,
                enable_tracing=enable_tracing,
                mode=mode,
                mcp_server_keys=mcp_server_keys,
                stream=False,
            ),
        )

    async def stream(
        self,
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
        event_streamer = getattr(self._chat_runtime_service, "astream_events", None)
        if not callable(event_streamer):
            raise AttributeError("Runtime service must expose astream_events(..., version='v3').")

        async for event in self._stream_v3_events(
            event_streamer=event_streamer,
            messages=messages,
            model_id=model_id,
            thread_id=thread_id,
            session_id=session_id,
            collection_name=collection_name,
            enable_reranker=enable_reranker,
            enable_tracing=enable_tracing,
            mode=mode,
            mcp_server_keys=mcp_server_keys,
        ):
            yield event

    async def _stream_v3_events(
        self,
        *,
        event_streamer: Any,
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
        input_payload = {"messages": [message.model_dump() for message in messages]}
        config: dict[str, object] = {
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
        raw_stream = event_streamer(input_payload, config=config, version="v3")
        if inspect.isawaitable(raw_stream):
            raw_stream = await raw_stream

        async for raw_event in raw_stream:
            for event in _convert_v3_event(raw_event):
                yield event


def _convert_v3_event(raw_event: object) -> list[dict[str, object]]:
    if not isinstance(raw_event, dict):
        return []
    method = raw_event.get("method")
    params = raw_event.get("params")
    if not isinstance(params, dict):
        return []
    data = params.get("data")
    if method == "messages":
        return _convert_v3_message_event(data)
    if method == "tool_calls":
        safe_data = data if isinstance(data, dict) else {"event": str(data)}
        return [{"type": "tool_event", "data": cast(dict[str, object], safe_data)}]
    if method == "custom":
        return _convert_v3_custom_event(data)
    return []


def _convert_v3_message_event(data: object) -> list[dict[str, object]]:
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


def _convert_v3_custom_event(data: object) -> list[dict[str, object]]:
    if not isinstance(data, dict):
        return []
    event_type = data.get("type")
    if event_type not in {"references", "tool_event"}:
        return []
    payload = data.get("data")
    if not isinstance(payload, dict):
        payload = {}
    return [{"type": str(event_type), "data": cast(dict[str, object], payload)}]

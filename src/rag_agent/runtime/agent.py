"""Runtime agent facade for thread/run APIs.

This module centralizes request-shape normalization and delegates execution to
``ChatRuntimeService``. It is intentionally framework-agnostic so route modules
can reuse the same invocation and streaming behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

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
                    normalized.append(ChatMessage(role=role_raw, content=content_raw))
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
        return await self._chat_runtime_service.run_chat(
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
        async for event in self._chat_runtime_service.stream_chat(
            messages=[message.model_dump() for message in messages],
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

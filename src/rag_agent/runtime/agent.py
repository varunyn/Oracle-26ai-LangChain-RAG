"""Runtime request helpers for thread/run APIs.

This module centralizes request-shape normalization. It is intentionally
framework-agnostic so route modules can reuse the same message behavior.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from api.schemas import ChatMessage


def normalize_messages(
    messages: list[dict[str, Any]] | None, message: str | None
) -> list[ChatMessage]:
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

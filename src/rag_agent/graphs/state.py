from __future__ import annotations

from typing import Literal, TypedDict

Mode = Literal["direct", "rag", "mcp", "mixed"]


class ChatGraphContext(TypedDict, total=False):
    model_id: str
    collection_name: str
    mode: Mode
    enable_reranker: bool
    enable_tracing: bool
    mcp_server_keys: list[str]


class ChatGraphState(TypedDict, total=False):
    messages: list[dict[str, object]]
    context: ChatGraphContext
    references: dict[str, object]

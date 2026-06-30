from __future__ import annotations

from typing import Annotated, Literal

from typing_extensions import TypedDict

from src.rag_agent.runtime.memory import merge_chat_messages

Mode = Literal["direct", "rag", "mcp", "mixed"]


class ChatGraphContext(TypedDict, total=False):
    model_id: str
    session_id: str
    collection_name: str
    mode: Mode
    enable_reranker: bool
    enable_tracing: bool
    mcp_server_keys: list[str]


class ChatGraphState(TypedDict, total=False):
    messages: Annotated[list[object], merge_chat_messages]
    context: ChatGraphContext
    references: dict[str, object]
    progress: str

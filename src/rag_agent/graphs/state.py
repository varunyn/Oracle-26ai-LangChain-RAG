from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Mode = Literal["direct", "rag", "mcp", "mixed"]


class ChatGraphContext(TypedDict, total=False):
    model_id: str
    session_id: str
    request_id: str
    thread_id: str
    user_id: str
    release: str
    collection_name: str
    mode: Mode
    enable_reranker: bool
    enable_tracing: bool
    max_rounds: int
    mcp_server_keys: list[str]
    mcp_config_digest: str


class ChatGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: ChatGraphContext
    references: dict[str, object]
    progress: str
    mixed_result: dict[str, object]
    mixed_state_messages: list[object]


class MCPSubGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int

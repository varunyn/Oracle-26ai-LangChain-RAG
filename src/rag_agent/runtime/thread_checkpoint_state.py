from __future__ import annotations

from typing import Any, TypedDict


class ThreadCheckpointState(TypedDict, total=False):
    """State shape persisted for chat runtime thread compatibility."""

    messages: list[Any]
    final_answer: str
    error: str | None
    standalone_question: str | None
    citations: list[dict[str, Any]]
    reranker_docs: list[dict[str, Any]]
    context_usage: dict[str, Any] | None
    mcp_used: bool
    mcp_tools_used: list[str]
    mcp_tool_invocations: list[dict[str, Any]]
    model_id: str
    usage: dict[str, Any]
    cost_usd: float | None


from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.rag_agent.graphs.runtime import references_from_result, result_to_assistant_message

logger = logging.getLogger(__name__)


def merge_references(mode: str, result: dict[str, object]) -> dict[str, object]:
    references = references_from_result(result, mode=mode)
    error = result.get("error")
    if isinstance(error, dict):
        references["error"] = error
    return references


def assistant_message_from_result(mode: str, result: dict[str, object]) -> AIMessage:
    return result_to_assistant_message(mode, result)


def assistant_message_from_exception(mode: str, exc: Exception) -> AIMessage:
    logger.exception("LangGraph %s node failed", mode)
    return assistant_message_from_result(
        mode,
        {
            "final_answer": (
                "I couldn't complete the request because the runtime "
                "backend returned an error. Please try again after the backend "
                "connection is healthy."
            ),
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        },
    )

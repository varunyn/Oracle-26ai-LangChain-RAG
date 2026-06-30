"""Expose the app's RAG runtime as MCP tools."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import CodeMode
from pydantic import Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.settings import Settings, get_settings
from src.rag_agent.graphs.chat_agent import chat_agent
from src.rag_agent.infrastructure.mcp_settings import normalize_mcp_transport

logger = logging.getLogger(__name__)
mcp = FastMCP(
    "RAG as MCP server (LangChain workflow)",
    transforms=[CodeMode()],
)


def _get_settings() -> Settings:
    return get_settings()


def _build_rag_config(
    collection_name: str | None = None,
    enable_reranker: bool | None = None,
) -> dict[str, object]:
    """Build the context used for RAG-only graph execution."""
    settings = _get_settings()
    return {
        "model_id": settings.LLM_MODEL_ID,
        "enable_reranker": (
            enable_reranker if enable_reranker is not None else settings.ENABLE_RERANKER
        ),
        "collection_name": collection_name or settings.DEFAULT_COLLECTION,
        "thread_id": str(uuid.uuid4()),
        "mode": "rag",
    }


def _run_server(transport: str) -> None:
    settings = _get_settings()
    if transport == "stdio":
        mcp.run(transport=cast(Any, transport))
        return
    mcp.run(
        transport=cast(Any, transport),
        host=settings.HOST,
        port=settings.PORT,
        log_level="INFO",
    )


@mcp.tool
def rag_ask(
    question: Annotated[
        str,
        Field(description="Question to answer using the RAG workflow."),
    ],
    collection_name: Annotated[
        str | None,
        Field(
            description="Vector-store collection or table name. Defaults to the configured collection."
        ),
    ] = None,
    enable_reranker: Annotated[
        bool,
        Field(description="Whether to rerank retrieved chunks before answering."),
    ] = True,
) -> dict[str, object]:
    """Return an answer, citations, and optional error from the RAG workflow."""
    question_text = question.strip()
    if not question_text:
        return {"answer": "", "citations": [], "error": "Empty question."}

    run_config = _build_rag_config(
        collection_name=collection_name,
        enable_reranker=enable_reranker,
    )

    try:
        final_state = chat_agent.invoke(
            {"messages": [{"role": "user", "content": question_text}]},
            config={"configurable": {"thread_id": run_config["thread_id"]}},
            context={
                "model_id": run_config["model_id"],
                "collection_name": run_config["collection_name"],
                "enable_reranker": run_config["enable_reranker"],
                "mode": "rag",
            },
        )
    except Exception as exc:
        logger.exception("RAG invoke error in MCP")
        return {"answer": "", "citations": [], "error": str(exc)}

    answer = str(final_state.get("final_answer") or "").strip()
    citations_raw_obj = final_state.get("citations")
    citations_raw = citations_raw_obj if isinstance(citations_raw_obj, list) else []
    citations = [
        {
            "source": str(citation.get("source", "")),
            "page": str(citation.get("page", "")),
        }
        for citation in citations_raw
        if isinstance(citation, Mapping)
    ]
    return {
        "answer": answer,
        "citations": citations,
        "error": final_state.get("error"),
    }


if __name__ == "__main__":
    from src.rag_agent.utils.logging_config import setup_logging

    setup_logging()
    transport = normalize_mcp_transport(_get_settings().TRANSPORT)
    if transport not in {"stdio", "streamable-http"}:
        raise RuntimeError(f"Unsupported TRANSPORT: {transport}")
    _run_server(transport)

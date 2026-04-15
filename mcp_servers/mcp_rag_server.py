"""Expose the app's RAG runtime as MCP tools."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, cast

from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import CodeMode
from pydantic import Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.services.graph_service import ChatRuntimeService
from api.settings import Settings, get_settings
from src.rag_agent.infrastructure.mcp_settings import normalize_mcp_transport

logger = logging.getLogger(__name__)
mcp = FastMCP(
    "RAG as MCP server (LangChain workflow)",
    transforms=[CodeMode()],
)
_GRAPH_SERVICE = ChatRuntimeService()


def _get_settings() -> Settings:
    return get_settings()


def _build_rag_config(
    collection_name: str | None = None,
    enable_reranker: bool | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    """Build the configurable payload used for RAG-only workflow execution."""
    settings = _get_settings()
    return {
        "model_id": model_id or settings.LLM_MODEL_ID,
        "embed_model_type": settings.EMBED_MODEL_TYPE,
        "enable_reranker": (
            enable_reranker if enable_reranker is not None else settings.ENABLE_RERANKER
        ),
        "collection_name": collection_name or settings.DEFAULT_COLLECTION,
        "thread_id": str(uuid.uuid4()),
        "mode": "rag",
        "max_rounds": getattr(settings, "MCP_MAX_ROUNDS", 2),
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
        final_state = asyncio.run(
            _GRAPH_SERVICE.run_chat(
                messages=[{"role": "user", "content": question_text}],
                model_id=cast(str | None, run_config.get("model_id")),
                thread_id=cast(str | None, run_config.get("thread_id")),
                session_id=None,
                collection_name=cast(str | None, run_config.get("collection_name")),
                enable_reranker=cast(bool | None, run_config.get("enable_reranker")),
                enable_tracing=None,
                mode="rag",
                mcp_server_keys=None,
                stream=False,
            )
        )
    except Exception as exc:
        logger.exception("RAG invoke error in MCP")
        return {"answer": "", "citations": [], "error": str(exc)}

    answer = (final_state.get("final_answer") or "").strip()
    citations_raw = final_state.get("citations") or []
    citations = [{"source": c.get("source", ""), "page": c.get("page", "")} for c in citations_raw]
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

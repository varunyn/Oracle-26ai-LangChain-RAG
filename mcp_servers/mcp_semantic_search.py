"""Expose semantic search MCP tools backed by the app's retrieval stack."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

from fastmcp import FastMCP
from pydantic import Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.settings import Settings, get_settings
from src.rag_agent.infrastructure.db_utils import (
    get_connection,
    list_collections,
    list_sources_in_collection,
)
from src.rag_agent.infrastructure.mcp_settings import normalize_mcp_transport
from src.rag_agent.infrastructure.oci_models import get_embedding_model
from src.rag_agent.infrastructure.retrieval import normalize_search_mode, search_documents
from src.rag_agent.utils.utils import docs_serializable

logger = logging.getLogger(__name__)
mcp = FastMCP("Semantic Search MCP server")


def _get_settings() -> Settings:
    return get_settings()


def _default_collection_name() -> str:
    settings = _get_settings()
    collections = settings.COLLECTION_LIST
    if collections:
        return collections[0]
    return settings.DEFAULT_COLLECTION


def _run_server(transport: str) -> None:
    settings = _get_settings()
    if transport == "stdio":
        mcp.run(transport=cast(Any, transport), log_level="INFO")
        return
    mcp.run(
        transport=cast(Any, transport),
        host=settings.HOST,
        port=settings.PORT,
        log_level="INFO",
    )


@mcp.tool
def semantic_search(
    query: Annotated[str, Field(description="The search query to find relevant documents.")],
    top_k: Annotated[int, Field(description="Number of results to return.")] = 5,
    collection_name: Annotated[
        str | None,
        Field(
            description="Collection or table name to search. Defaults to the first configured collection."
        ),
    ] = None,
    search_mode: Annotated[
        str | None,
        Field(description="Retrieval mode: vector, hybrid, or text."),
    ] = None,
) -> Mapping[str, object]:
    """Return retrieval results for the given query."""
    settings = _get_settings()
    effective_collection = collection_name or _default_collection_name()
    effective_mode = normalize_search_mode(search_mode or settings.MCP_SEARCH_MODE)

    try:
        embed_model = get_embedding_model(settings.EMBED_MODEL_TYPE)
        with get_connection() as conn:
            relevant_docs = search_documents(
                conn=conn,
                collection_name=effective_collection,
                embed_model=embed_model,
                query=query,
                top_k=top_k,
                search_mode=effective_mode,
            )
    except Exception as exc:
        logger.exception("MCP semantic search failed")
        return {"error": str(exc)}

    logger.info(
        "Result from MCP retrieval search: mode=%s docs=%d",
        effective_mode,
        len(relevant_docs),
    )
    if settings.DEBUG:
        logger.debug("Relevant docs detail: %s", relevant_docs)
    return {"relevant_docs": docs_serializable(relevant_docs)}


@mcp.tool
def get_collections() -> list[str]:
    """Return the configured vector-store collection names."""
    return list_collections()


@mcp.tool
def list_documents_in_collection(
    collection_name: Annotated[
        str | None,
        Field(
            description="Collection or table name to inspect. Defaults to the first configured collection."
        ),
    ] = None,
) -> list[tuple[str | None, int]]:
    """Return unique document sources and chunk counts for a collection."""
    effective_collection = collection_name or _default_collection_name()
    try:
        return list_sources_in_collection(effective_collection)
    except Exception:
        logger.exception("Failed to list documents in collection %s", effective_collection)
        return []


if __name__ == "__main__":
    from src.rag_agent.utils.logging_config import setup_logging

    setup_logging()
    transport = normalize_mcp_transport(_get_settings().TRANSPORT)
    if transport not in {"stdio", "streamable-http"}:
        raise RuntimeError(f"Unsupported TRANSPORT: {transport}")
    _run_server(transport)

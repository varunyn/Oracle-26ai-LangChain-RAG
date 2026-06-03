from __future__ import annotations

import logging
from typing import cast

import oracledb
from langchain_core.documents import Document
from langchain_oci import OCIGenAIEmbeddings  # type: ignore[import-untyped]

from .oci_models import get_oracle_vs

logger = logging.getLogger(__name__)

_ALLOWED_SEARCH_MODES = {"vector"}
_ORACLE_SIMPLE_IDENTIFIER_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$#"
)


def normalize_search_mode(raw_mode: str | None, default: str = "vector") -> str:
    mode = (raw_mode or default or "vector").strip().lower()
    if mode in _ALLOWED_SEARCH_MODES:
        return mode
    logger.warning("Unsupported search mode '%s'; falling back to 'vector'", mode)
    return "vector"


def normalize_collection_name(collection_name: str) -> str:
    normalized = collection_name.strip()
    if not normalized:
        raise ValueError("collection_name is required")
    if normalized.startswith('"') and normalized.endswith('"'):
        return normalized
    if any(char not in _ORACLE_SIMPLE_IDENTIFIER_CHARS for char in normalized):
        return normalized
    return normalized.upper()


def _matches_metadata_filters(doc: Document, metadata_filters: dict[str, object]) -> bool:
    for key, expected in metadata_filters.items():
        if doc.metadata.get(key) != expected:
            return False
    return True


def _apply_metadata_filters(
    docs: list[Document], metadata_filters: dict[str, object] | None
) -> list[Document]:
    if not metadata_filters:
        return docs
    return [doc for doc in docs if _matches_metadata_filters(doc, metadata_filters)]


def search_documents(
    conn: oracledb.Connection,
    collection_name: str,
    embed_model: OCIGenAIEmbeddings,
    query: str,
    top_k: int,
    search_mode: str,
    metadata_filters: dict[str, object] | None = None,
) -> list[Document]:
    k = max(1, int(top_k))
    _ = normalize_search_mode(search_mode)
    v_store = get_oracle_vs(
        conn=conn,
        collection_name=normalize_collection_name(collection_name),
        embed_model=embed_model,
    )
    docs = cast(
        list[Document],
        v_store.similarity_search(query=query, k=k, filter=metadata_filters),
    )
    return docs[:k]

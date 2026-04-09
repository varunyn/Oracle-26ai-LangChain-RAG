from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

import oracledb
from langchain_core.documents import Document
from langchain_oci import OCIGenAIEmbeddings

from .oci_models import get_oracle_vs

logger = logging.getLogger(__name__)

_ALLOWED_SEARCH_MODES = {"vector", "text", "hybrid"}
_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def normalize_search_mode(raw_mode: str | None, default: str = "vector") -> str:
    mode = (raw_mode or default or "vector").strip().lower()
    if mode in _ALLOWED_SEARCH_MODES:
        return mode
    logger.warning("Unknown search mode '%s'; falling back to '%s'", mode, default)
    return default


def _safe_table_name(table_name: str) -> str:
    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"Invalid collection/table name: {table_name!r}")
    return table_name


def _metadata_to_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, oracledb.LOB):
        value = value.read()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def _text_search_docs(
    conn: oracledb.Connection,
    collection_name: str,
    query: str,
    limit: int,
) -> list[Document]:
    sanitized_collection = _safe_table_name(collection_name)
    query_clean = query.strip().lower()
    if not query_clean:
        return []

    sql = (
        f"SELECT text, metadata FROM {sanitized_collection} "
        "WHERE DBMS_LOB.INSTR(LOWER(text), :needle) > 0 "
        f"FETCH FIRST {max(1, int(limit))} ROWS ONLY"
    )

    docs: list[Document] = []
    with conn.cursor() as cursor:
        cursor.execute(sql, {"needle": query_clean})
        for row in cursor.fetchall():
            text_value = row[0]
            if isinstance(text_value, oracledb.LOB):
                page_content = text_value.read()
            else:
                page_content = str(text_value or "")
            if isinstance(page_content, bytes):
                page_content = page_content.decode("utf-8", errors="replace")
            metadata = _metadata_to_dict(row[1] if len(row) > 1 else None)
            docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


def _doc_key(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    chunk_offset = str(doc.metadata.get("chunk_offset", ""))
    return f"{source}::{chunk_offset}::{doc.page_content[:180]}"


def _rrf_fuse(vector_docs: list[Document], text_docs: list[Document], top_k: int) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    rrf_k = 60.0

    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)
        doc_map.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    for rank, doc in enumerate(text_docs, start=1):
        key = _doc_key(doc)
        doc_map.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_map[key] for key, _ in ordered[: max(1, int(top_k))]]


def search_documents(
    conn: oracledb.Connection,
    collection_name: str,
    embed_model: OCIGenAIEmbeddings,
    query: str,
    top_k: int,
    search_mode: str,
) -> list[Document]:
    mode = normalize_search_mode(search_mode)
    k = max(1, int(top_k))

    if mode == "text":
        return _text_search_docs(conn, collection_name, query, k)

    v_store = get_oracle_vs(conn=conn, collection_name=collection_name, embed_model=embed_model)
    if mode == "vector":
        return cast(list[Document], v_store.similarity_search(query=query, k=k))

    vector_docs = cast(list[Document], v_store.similarity_search(query=query, k=max(k * 2, 6)))
    text_docs = _text_search_docs(conn, collection_name, query, max(k * 2, 6))
    if not vector_docs:
        return text_docs[:k]
    if not text_docs:
        return vector_docs[:k]
    return _rrf_fuse(vector_docs, text_docs, k)

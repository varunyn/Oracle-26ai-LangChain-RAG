"""Shared citation normalization utilities for RAG and mixed chat responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from langchain_core.documents import Document


def normalize_citation(raw: Mapping[str, object]) -> dict[str, object]:
    """Normalize a citation object into canonical source/page/link fields."""
    source = (
        raw.get("source")
        or raw.get("title")
        or raw.get("document_name")
        or raw.get("name")
        or raw.get("file_name")
        or raw.get("id")
        or raw.get("link")
        or raw.get("url")
        or raw.get("source_url")
        or raw.get("file_path")
        or ""
    )
    page = raw.get("page") or raw.get("page_number")
    link = raw.get("link") or raw.get("url") or raw.get("source_url")

    source_text = str(source).strip()
    page_text = str(page).strip() if page is not None else ""
    link_text = str(link).strip() if link is not None else ""

    normalized: dict[str, object] = {
        "source": source_text,
        "page": page_text or None,
        "link": link_text or None,
    }
    if "score" in raw:
        normalized["score"] = raw.get("score")
    return normalized


def normalize_citations(raw_citations: Sequence[object]) -> list[dict[str, object]]:
    """Normalize a sequence of citation-like objects."""
    normalized: list[dict[str, object]] = []
    for item in raw_citations:
        if isinstance(item, Mapping):
            normalized.append(normalize_citation(cast(Mapping[str, object], item)))
    return normalized


def citations_from_documents(docs: Sequence[Document]) -> list[dict[str, object]]:
    """Build normalized citation payloads from LangChain documents."""
    citations: list[dict[str, object]] = []
    for doc in docs:
        metadata = cast(Mapping[str, object], dict(doc.metadata or {}))
        citations.append(normalize_citation(metadata))
    return citations

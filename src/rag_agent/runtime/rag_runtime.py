from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool

from src.rag_agent.core.citations import citations_from_documents
from src.rag_agent.infrastructure.db_utils import get_pooled_connection
from src.rag_agent.infrastructure.oci_models import (
    get_embedding_model,
    get_llm,
)
from src.rag_agent.infrastructure.oci_models import (
    rerank_documents as oci_rerank_documents,
)
from src.rag_agent.infrastructure.retrieval import search_documents
from src.rag_agent.prompts.runtime_agents import RAG_ANSWER_PROMPT_TEMPLATE

from .llm_invocation import invoke_llm_with_optional_config
from .observability import extract_usage

logger = logging.getLogger(__name__)


def build_oracle_retrieval_tool(
    *,
    collection_name: str | None,
    filter_docs: Callable[[str, list[Document]], list[Document]],
) -> StructuredTool:
    state: dict[str, object] = {"docs": []}
    collection = collection_name or "RAG_KNOWLEDGE_BASE"

    def retrieve_context(query: str) -> tuple[str, list[Document]]:
        """Retrieve Oracle knowledge-base and documentation context for a user question."""
        try:
            with get_pooled_connection() as conn:
                embed_model = get_embedding_model()
                docs = cast(
                    list[Document],
                    search_documents(
                        conn=conn,
                        collection_name=collection,
                        embed_model=embed_model,
                        query=query,
                        top_k=8,
                        search_mode="vector",
                    ),
                )
            filtered = filter_docs(query, docs)
            state["docs"] = filtered
            state.pop("error", None)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in filtered
            )
            return serialized, filtered
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            logger.exception(
                "oracle_retrieval_failed collection=%s query_len=%d",
                collection,
                len(query or ""),
            )
            state["docs"] = []
            state["error"] = message
            return (
                "Oracle retrieval failed while searching the knowledge base. "
                f"Error: {message}",
                [],
            )

    tool = StructuredTool.from_function(
        retrieve_context,
        name="oracle_retrieval",
        description="Retrieve Oracle knowledge-base and documentation context for a user question.",
        response_format="content_and_artifact",
    )
    setattr(tool, "_retrieval_state", state)
    return tool


def retrieve_oracle_docs(*, query: str, collection_name: str | None, k: int) -> list[Document]:
    collection = collection_name or "RAG_KNOWLEDGE_BASE"
    from api.settings import get_settings

    search_mode = str(get_settings().RAG_SEARCH_MODE or "vector").strip().lower()

    with get_pooled_connection() as conn:
        embed_model = get_embedding_model()
        docs = cast(
            list[Document],
            search_documents(
                conn=conn,
                collection_name=collection,
                embed_model=embed_model,
                query=query,
                top_k=k,
                search_mode=search_mode,
            ),
        )
        if docs:
            logger.info(
                "rag_retrieval mode=%s collection=%s docs=%d",
                search_mode,
                collection,
                len(docs),
            )
            return docs

    logger.warning("rag_retrieval_no_docs collection=%s query_len=%d", collection, len(query or ""))
    return []


async def synthesize_rag_answer(
    *,
    question: str,
    docs: list[Document],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> tuple[str, dict[str, int] | None, str]:
    context = format_retrieved_docs(docs)
    prompt = RAG_ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)
    answer_messages = [HumanMessage(content=prompt)]
    llm = get_llm(model_id=model_id)
    final_message = await asyncio.to_thread(
        invoke_llm_with_optional_config,
        llm,
        answer_messages,
        run_config,
    )
    resolved_model_id = cast(str | None, getattr(llm, "model_id", None)) or model_id or "unknown"
    return (
        str(getattr(final_message, "content", "") or "").strip(),
        extract_usage(final_message),
        resolved_model_id,
    )


def filter_retrieved_docs(query: str, docs: list[Document]) -> list[Document]:
    terms = query_terms(query)
    if not docs or not terms:
        return docs

    required_overlap = 2 if len(terms) >= 3 else 1
    scored: list[tuple[int, Document]] = []
    for doc in docs:
        text_blob = " ".join(
            [
                str(doc.page_content or ""),
                str((doc.metadata or {}).get("source") or ""),
                str((doc.metadata or {}).get("title") or ""),
                str((doc.metadata or {}).get("file_name") or ""),
            ]
        ).lower()
        overlap = sum(1 for term in terms if term in text_blob)
        scored.append((overlap, doc))

    kept = [doc for overlap, doc in scored if overlap >= required_overlap]
    if kept:
        return kept[:5]

    best_overlap = max((overlap for overlap, _ in scored), default=0)
    if best_overlap > 0:
        best_docs = [doc for overlap, doc in scored if overlap == best_overlap]
        return best_docs[:3]

    return []


def rerank_retrieved_docs(
    query: str,
    docs: list[Document],
    *,
    enable_reranker: bool | None,
) -> list[Document]:
    if enable_reranker is not True:
        return docs
    try:
        return cast(list[Document], oci_rerank_documents(query, docs))
    except Exception:
        logger.exception("oci_rerank_failed docs=%d query_len=%d", len(docs), len(query or ""))
        return filter_retrieved_docs(query, docs)


def query_terms(query: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "that",
        "the",
        "to",
        "use",
        "what",
        "with",
    }
    terms = re.findall(r"[a-zA-Z0-9_]+", (query or "").lower())
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < 3 or term in stopwords or term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def serialize_docs(docs: list[Document]) -> list[dict[str, object]]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in docs
    ]


def citations_from_docs(docs: list[Document]) -> list[dict[str, object]]:
    return citations_from_documents(docs)


def format_retrieved_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents were found."
    return "\n\n".join(f"[{idx}] {doc.page_content}" for idx, doc in enumerate(docs, start=1))


__all__ = [
    "build_oracle_retrieval_tool",
    "citations_from_docs",
    "filter_retrieved_docs",
    "format_retrieved_docs",
    "query_terms",
    "rerank_retrieved_docs",
    "retrieve_oracle_docs",
    "serialize_docs",
    "synthesize_rag_answer",
]

"""Oracle/OCI adapter for the transport-neutral knowledge service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.documents import Document

from ..application.oracle_knowledge import OracleKnowledgeService, RetrievalCandidate
from .db_utils import get_connection, list_sources_in_collection
from .oci_models import (
    get_embedding_model,
    get_oracle_vs,
    rerank_documents_with_scores,
)
from .retrieval import normalize_collection_name


class KnowledgeReadinessProbe:
    """Injected conservative readiness checks; never constructs a chat model."""

    def __init__(
        self,
        settings: object,
        *,
        oracle_check=None,
        embedding_check=None,
        cache_seconds: float = 15.0,
    ):
        self.settings = settings
        self.oracle_check = oracle_check or self._check_oracle
        self.embedding_check = embedding_check or self._check_embedding
        self.cache_seconds = cache_seconds
        self._cached = None
        self._cached_at = 0.0

    def _check_oracle(self) -> bool:
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT 1 FROM dual")
                return True
            finally:
                cursor.close()

    def _check_embedding(self) -> bool:
        model = get_embedding_model(str(getattr(self.settings, "EMBED_MODEL_TYPE", "OCI")))
        model.embed_query("oracle knowledge readiness")
        return True

    def check(self) -> tuple[bool, str]:
        import time

        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self.cache_seconds:
            return self._cached
        try:
            mapping = getattr(self.settings, "ORACLE_KNOWLEDGE_BASES", {})
            default = getattr(self.settings, "ORACLE_KNOWLEDGE_DEFAULT_KEY", "")
            if not mapping or default not in mapping:
                result = (False, "configuration unavailable")
            elif not self.oracle_check():
                result = (False, "oracle unavailable")
            elif not self.embedding_check():
                result = (False, "embedding unavailable")
            else:
                result = (True, "ready")
        except Exception:
            result = (False, "readiness unavailable")
        self._cached, self._cached_at = result, now
        return result

    async def check_async(self) -> tuple[bool, str]:
        import asyncio

        return await asyncio.to_thread(self.check)


class OracleKnowledgeAdapter:
    """Small provider adapter; no chat model or answer generation is involved."""

    def __init__(self, settings: object) -> None:
        self._settings = settings
        self._embedder = get_embedding_model(str(settings.EMBED_MODEL_TYPE))

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embedder.embed_query(text)

    def retrieve(
        self,
        collection: str,
        query_embedding: Sequence[float],
        limit: int,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievalCandidate]:
        with get_connection() as conn:
            store = get_oracle_vs(conn, normalize_collection_name(collection), self._embedder)
            search_kwargs: dict[str, object] = {"k": limit}
            if metadata_filters:
                search_kwargs["filter"] = dict(metadata_filters)
            scored_docs = store.similarity_search_by_vector_with_relevance_scores(
                list(query_embedding), **search_kwargs
            )
        return [
            RetrievalCandidate(doc.page_content, dict(doc.metadata or {}), float(score))
            for doc, score in scored_docs
        ]

    def list_documents(self, collection: str) -> list[Mapping[str, object]]:
        return [
            {"source": source, "page_count": count}
            for source, count in list_sources_in_collection(collection)
        ]

    def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> list[tuple[RetrievalCandidate, float | None]]:
        docs = [Document(page_content=c.content, metadata=dict(c.metadata)) for c in candidates]
        ranked = rerank_documents_with_scores(query, docs)
        return [
            (next(c for c, d in zip(candidates, docs) if d is ranked_doc), score)
            for ranked_doc, score in ranked
        ]


def build_oracle_knowledge_service(
    settings: object,
    *,
    enable_reranker: bool | None = None,
    collection_name: str | None = None,
    knowledge_bases: Mapping[str, str] | None = None,
    default_key: str | None = None,
) -> OracleKnowledgeService:
    """Construct the shared service for MCP and existing chat retrieval callers."""
    adapter = OracleKnowledgeAdapter(settings)
    selected_mapping = (
        {
            "chat": collection_name
            or str(getattr(settings, "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE"))
        }
        if knowledge_bases is None
        else dict(knowledge_bases)
    )
    return OracleKnowledgeService(
        knowledge_bases=selected_mapping,
        default_knowledge_base=default_key
        or ("chat" if knowledge_bases is None else next(iter(selected_mapping), None)),
        embedder=adapter,
        retriever=adapter,
        reranker=adapter,
        enable_reranker=bool(
            getattr(settings, "ORACLE_KNOWLEDGE_ENABLE_RERANKER", True)
            if enable_reranker is None
            else enable_reranker
        ),
        allow_reranker_override=bool(
            getattr(settings, "ORACLE_KNOWLEDGE_ALLOW_RERANKER_OVERRIDE", True)
        ),
        candidate_limit=int(getattr(settings, "ORACLE_KNOWLEDGE_CANDIDATE_LIMIT", 20)),
        max_query_length=int(getattr(settings, "ORACLE_KNOWLEDGE_MAX_QUERY_LENGTH", 8192)),
        max_result_limit=int(getattr(settings, "ORACLE_KNOWLEDGE_MAX_RESULT_LIMIT", 50)),
        max_metadata_filters=int(getattr(settings, "ORACLE_KNOWLEDGE_MAX_METADATA_FILTERS", 8)),
        max_candidate_limit=int(getattr(settings, "ORACLE_KNOWLEDGE_MAX_CANDIDATE_LIMIT", 100)),
        execution_timeout_seconds=float(
            getattr(settings, "ORACLE_KNOWLEDGE_TIMEOUT_SECONDS", 30.0)
        ),
    )

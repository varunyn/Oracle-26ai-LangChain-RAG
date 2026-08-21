"""Typed, transport-neutral Oracle knowledge retrieval application service."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.citations import normalize_citation
from ..utils.logging_config import get_request_id

logger = logging.getLogger(__name__)
CONTRACT_VERSION = "1.0"
Outcome = Literal["success", "no_hits", "invalid_request", "forbidden", "backend_error"]
RerankingStatus = Literal["disabled", "applied", "failed"]


class SearchKnowledgeRequest(BaseModel):
    """Validated public search input; collection names are intentionally absent."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=100000)
    knowledge_base: str | None = Field(default=None, min_length=1, max_length=128)
    limit: int = Field(default=5, ge=1, le=100)
    candidate_limit: int | None = Field(default=None, ge=1, le=100)
    rerank: bool | None = None
    metadata_filters: dict[str, str | int | bool] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def non_blank_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    chunk_id: str
    content: str
    rank: int
    source: str
    title: str | None = None
    page: str | None = None
    link: str | None = None
    retrieval_score: float | None = None
    reranking_score: float | None = None


class SearchKnowledgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = CONTRACT_VERSION
    outcome: Outcome
    query: str
    knowledge_base: str | None
    reranking_status: RerankingStatus
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str


class KnowledgeBaseListResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = CONTRACT_VERSION
    outcome: Outcome
    knowledge_bases: list[KnowledgeBase] = Field(default_factory=list)
    error: str | None = None


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    source: str
    title: str | None = None
    page_count: int | None = None


class ListDocumentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = CONTRACT_VERSION
    outcome: Outcome
    knowledge_base: str | None
    documents: list[DocumentSummary] = Field(default_factory=list)
    error: str | None = None


class InternalRetrievalResult(BaseModel):
    """Internal chat seam retaining provider candidates and explicit failures."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    outcome: Outcome
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    content: str
    metadata: Mapping[str, object]
    retrieval_score: float | None = None
    provider_id: str | None = None


class RetrievalCandidateProtocol(Protocol):
    content: str
    metadata: Mapping[str, object]
    retrieval_score: float | None


class Embedder(Protocol):
    def embed_query(self, text: str) -> Sequence[float]: ...


class Retriever(Protocol):
    def retrieve(
        self,
        collection: str,
        query_embedding: Sequence[float],
        limit: int,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> Sequence[RetrievalCandidate]: ...

    def list_documents(self, collection: str) -> Sequence[Mapping[str, object]]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> Sequence[tuple[RetrievalCandidate, float | None]]: ...


class OracleKnowledgeService:
    """Own validation, friendly-key resolution, retrieval, reranking, and shaping."""

    def __init__(
        self,
        *,
        knowledge_bases: Mapping[str, str],
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker | None = None,
        allowed_metadata_keys: frozenset[str] = frozenset({"source", "title", "page"}),
        candidate_limit: int = 20,
        enable_reranker: bool = True,
        default_knowledge_base: str | None = None,
        max_query_length: int = 100000,
        max_result_limit: int = 50,
        max_metadata_filters: int = 8,
        max_candidate_limit: int = 100,
        execution_timeout_seconds: float = 30.0,
        allow_reranker_override: bool = False,
    ) -> None:
        self._knowledge_bases = dict(knowledge_bases)
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._allowed_metadata_keys = allowed_metadata_keys
        self._candidate_limit = max(1, min(candidate_limit, 100))
        self._enable_reranker = enable_reranker
        if enable_reranker and reranker is None:
            raise ValueError("reranker is required when reranking is enabled")
        self._default_key = default_knowledge_base
        if self._default_key not in self._knowledge_bases:
            raise ValueError("default knowledge base must be an allowed key")
        self._max_query_length = max(1, max_query_length)
        self._max_result_limit = max(1, min(max_result_limit, 100))
        self._max_metadata_filters = max(0, max_metadata_filters)
        self._max_candidate_limit = max(1, min(max_candidate_limit, 100))
        self._execution_timeout_seconds = max(0.01, execution_timeout_seconds)
        self._allow_reranker_override = allow_reranker_override

    @property
    def knowledge_base_keys(self) -> tuple[str, ...]:
        """Friendly keys exposed by this service, in deterministic order."""
        return tuple(sorted(self._knowledge_bases))

    @property
    def max_query_length(self) -> int:
        return self._max_query_length

    @property
    def max_result_limit(self) -> int:
        return self._max_result_limit

    @property
    def max_candidate_limit(self) -> int:
        return self._max_candidate_limit

    @property
    def max_metadata_filters(self) -> int:
        return self._max_metadata_filters

    def list_knowledge_bases(self) -> KnowledgeBaseListResult:
        return KnowledgeBaseListResult(
            outcome="success",
            knowledge_bases=[KnowledgeBase(key=key) for key in sorted(self._knowledge_bases)],
        )

    def list_documents(self, knowledge_base: str | None = None) -> ListDocumentsResult:
        selected = knowledge_base or self._default_key
        if selected not in self._knowledge_bases:
            return ListDocumentsResult(
                outcome="forbidden", knowledge_base=None, error="knowledge base is not allowed"
            )
        try:
            docs = [
                dict(item)
                for item in self._retriever.list_documents(self._knowledge_bases[selected])
            ]
        except Exception as exc:
            logger.error(
                "Oracle knowledge failure: stage=document_discovery type=%s code=provider_failure",
                type(exc).__name__,
            )
            return ListDocumentsResult(
                outcome="backend_error",
                knowledge_base=selected,
                error="knowledge backend unavailable",
            )
        safe = [
            DocumentSummary(
                document_id=str(d.get("document_id") or d.get("id") or _stable_id(d)),
                source=str(d.get("source") or d.get("title") or ""),
                title=str(d["title"]) if d.get("title") else None,
                page_count=int(d["page_count"]) if isinstance(d.get("page_count"), int) else None,
            )
            for d in docs
        ]
        return ListDocumentsResult(
            outcome="success" if safe else "no_hits", knowledge_base=selected, documents=safe
        )

    async def search(self, request: SearchKnowledgeRequest) -> SearchKnowledgeResult:
        """Run async-capable providers under a hard deadline."""
        tracer = trace.get_tracer(__name__)
        started = time.monotonic()
        with tracer.start_as_current_span(
            "oracle.knowledge.search", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("oracle.knowledge.request_id", get_request_id())
            span.set_attribute(
                "oracle.knowledge.knowledge_base",
                self._public_knowledge_base(request.knowledge_base) or "",
            )
            span.set_attribute("oracle.knowledge.result_limit", request.limit)
            span.set_attribute("oracle.knowledge.timeout_seconds", self._execution_timeout_seconds)
            try:
                result = await asyncio.wait_for(
                    self._search_async_impl(request), self._execution_timeout_seconds
                )
                span.set_attribute("oracle.knowledge.outcome", result.outcome)
                span.set_attribute(
                    "oracle.knowledge.elapsed_ms", (time.monotonic() - started) * 1000
                )
                span.set_attribute("oracle.knowledge.output_count", len(result.evidence))
                return result
            except TimeoutError:
                span.set_attribute("oracle.knowledge.outcome", "backend_error")
                span.set_attribute("oracle.knowledge.error_stage", "total")
                span.set_attribute("oracle.knowledge.error_type", "TimeoutError")
                span.set_attribute("oracle.knowledge.error_code", "timeout")
                logger.error("Oracle knowledge search timed out: code=timeout")
                return SearchKnowledgeResult(
                    outcome="backend_error",
                    query=request.query,
                    knowledge_base=self._public_knowledge_base(request.knowledge_base),
                    reranking_status="disabled",
                    error="knowledge backend unavailable",
                )

    async def retrieve_candidates(
        self,
        query: str,
        *,
        knowledge_base: str,
        limit: int,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> InternalRetrievalResult:
        if knowledge_base not in self._knowledge_bases:
            return InternalRetrievalResult(
                outcome="forbidden", error="knowledge base is not allowed"
            )
        try:
            vector = await _provider_call(self._embedder.embed_query, query)
            candidates = await _provider_call(
                self._retriever.retrieve,
                self._knowledge_bases[knowledge_base],
                vector,
                limit,
                metadata_filters or {},
            )
            return InternalRetrievalResult(
                outcome="success" if candidates else "no_hits", candidates=list(candidates)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Oracle knowledge failure: stage=chat_retrieval type=%s code=provider_failure",
                type(exc).__name__,
            )
            return InternalRetrievalResult(
                outcome="backend_error", error="knowledge backend unavailable"
            )

    async def rerank_candidates(
        self, query: str, candidates: Sequence[RetrievalCandidate], *, enabled: bool
    ) -> tuple[list[tuple[RetrievalCandidate, float | None]], RerankingStatus]:
        if not enabled or not candidates:
            return [(candidate, None) for candidate in candidates], "disabled"
        try:
            return list(await _provider_call(self._reranker.rerank, query, candidates)), "applied"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            span = trace.get_current_span()
            span.set_attribute("oracle.knowledge.error_stage", "reranking")
            span.set_attribute("oracle.knowledge.error_type", type(exc).__name__)
            span.set_attribute("oracle.knowledge.error_code", "provider_failure")
            logger.error(
                "Oracle knowledge failure: stage=reranking type=%s code=provider_failure",
                type(exc).__name__,
            )
            return [(candidate, None) for candidate in candidates], "failed"

    async def _search_async_impl(self, request: SearchKnowledgeRequest) -> SearchKnowledgeResult:
        selected = request.knowledge_base or self._default_key
        public_selected = self._public_knowledge_base(request.knowledge_base)
        if (
            len(request.query) > self._max_query_length
            or request.limit > self._max_result_limit
            or (
                request.candidate_limit is not None
                and request.candidate_limit > self._max_candidate_limit
            )
            or len(request.metadata_filters) > self._max_metadata_filters
        ):
            return SearchKnowledgeResult(
                outcome="invalid_request",
                query=request.query,
                knowledge_base=public_selected,
                reranking_status="disabled",
                error="search request exceeds configured bounds",
            )
        if selected not in self._knowledge_bases:
            return SearchKnowledgeResult(
                outcome="forbidden",
                query=request.query,
                knowledge_base=None,
                reranking_status="disabled",
                error="knowledge base is not allowed",
            )
        if any(key not in self._allowed_metadata_keys for key in request.metadata_filters):
            return SearchKnowledgeResult(
                outcome="invalid_request",
                query=request.query,
                knowledge_base=selected,
                reranking_status="disabled",
                error="unsupported metadata filter",
            )
        if request.rerank is not None and not self._allow_reranker_override:
            return SearchKnowledgeResult(
                outcome="invalid_request",
                query=request.query,
                knowledge_base=selected,
                reranking_status="disabled",
                error="reranker override is not allowed",
            )
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "oracle.knowledge.embedding", record_exception=False, set_status_on_exception=False
            ) as span:
                span.set_attribute("oracle.knowledge.request_id", get_request_id())
                span.set_attribute("oracle.knowledge.query_length", len(request.query))
                vector = await _provider_call(self._embedder.embed_query, request.query)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._backend_error("embedding", exc, request)
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "oracle.knowledge.search_oracle",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                span.set_attribute("oracle.knowledge.request_id", get_request_id())
                candidates = list(
                    await _provider_call(
                        self._retriever.retrieve,
                        self._knowledge_bases[selected],
                        vector,
                        max(request.limit, request.candidate_limit or self._candidate_limit),
                        request.metadata_filters,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._backend_error("retrieval", exc, request)
        reranking_enabled = request.rerank if request.rerank is not None else self._enable_reranker
        status: RerankingStatus = "disabled"
        ranked: list[tuple[RetrievalCandidate, float | None]] = [(c, None) for c in candidates]
        if reranking_enabled and self._reranker and candidates:
            status = "failed"
            try:
                with trace.get_tracer(__name__).start_as_current_span(
                    "oracle.knowledge.reranking",
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    span.set_attribute("oracle.knowledge.request_id", get_request_id())
                    span.set_attribute("oracle.knowledge.input_count", len(candidates))
                    ranked = list(
                        await _provider_call(self._reranker.rerank, request.query, candidates)
                    )
                status = "applied"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                span = trace.get_current_span()
                span.set_attribute("oracle.knowledge.error_stage", "reranking")
                span.set_attribute("oracle.knowledge.error_type", type(exc).__name__)
                span.set_attribute("oracle.knowledge.error_code", "provider_failure")
                logger.error(
                    "Oracle knowledge reranking failed: stage=reranking type=%s code=provider_failure",
                    type(exc).__name__,
                )
        evidence = [
            self._evidence(c, i, score) for i, (c, score) in enumerate(ranked[: request.limit], 1)
        ]
        return SearchKnowledgeResult(
            outcome="success" if evidence else "no_hits",
            query=request.query,
            knowledge_base=selected,
            reranking_status=status,
            evidence=evidence,
        )

    def _backend_error(
        self, stage: str, exc: BaseException, request: SearchKnowledgeRequest
    ) -> SearchKnowledgeResult:
        logger.error(
            "Oracle knowledge failure: stage=%s type=%s code=provider_failure",
            stage,
            type(exc).__name__,
        )
        span = trace.get_current_span()
        span.set_attribute("oracle.knowledge.error_stage", stage)
        span.set_attribute("oracle.knowledge.error_type", type(exc).__name__)
        span.set_attribute("oracle.knowledge.error_code", "provider_failure")
        return SearchKnowledgeResult(
            outcome="backend_error",
            query=request.query,
            knowledge_base=self._public_knowledge_base(request.knowledge_base),
            reranking_status="disabled",
            error="knowledge backend unavailable",
        )

    def _public_knowledge_base(self, requested: str | None) -> str | None:
        """Return only an allowlisted friendly key for public results and telemetry."""
        selected = requested or self._default_key
        return selected if selected in self._knowledge_bases else None

    @staticmethod
    def _evidence(
        candidate: RetrievalCandidate, rank: int, reranking_score: float | None
    ) -> Evidence:
        metadata = dict(candidate.metadata)
        document_id = str(
            metadata.get("document_id")
            or metadata.get("doc_id")
            or metadata.get("source")
            or _stable_id(metadata)
        )
        chunk_id = str(
            metadata.get("chunk_id")
            or metadata.get("id")
            or _stable_id({"document": document_id, "content": candidate.content})
        )
        citation = normalize_citation(metadata)
        return Evidence(
            document_id=document_id,
            chunk_id=chunk_id,
            content=candidate.content,
            rank=rank,
            source=str(citation["source"]),
            title=str(metadata["title"]) if metadata.get("title") else None,
            page=citation["page"],
            link=citation["link"],
            retrieval_score=candidate.retrieval_score,
            reranking_score=reranking_score,
        )


def _stable_id(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


async def _provider_call(call: object, *args: object) -> object:
    """Invoke async providers directly; keep sync providers off the event loop."""
    if inspect.iscoroutinefunction(call):
        return await call(*args)  # type: ignore[operator]
    return await asyncio.to_thread(call, *args)  # type: ignore[arg-type]

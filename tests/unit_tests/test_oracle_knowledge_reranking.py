import asyncio
import logging
from dataclasses import dataclass

from fastmcp import Client

import rag_agent.application.oracle_knowledge as knowledge_module
from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from rag_agent.application.oracle_knowledge import OracleKnowledgeService, SearchKnowledgeRequest


@dataclass
class Candidate:
    content: str
    metadata: dict[str, object]
    retrieval_score: float | None = 0.1


class Embedder:
    def embed_query(self, query):
        return [1.0]


class Retriever:
    def __init__(self):
        self.limit = None

    def retrieve(self, collection, vector, limit, filters=None):
        self.limit = limit
        return [
            Candidate("a", {"source": "a"}),
            Candidate("b", {"source": "b"}),
            Candidate("c", {"source": "c"}),
        ]

    def list_documents(self, collection):
        return []


class Reranker:
    def rerank(self, query, candidates):
        return [(candidates[2], 0.9), (candidates[0], 0.5), (candidates[1], 0.1)]


def test_reranking_status_order_scores_and_bounds() -> None:
    retriever = Retriever()
    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        default_knowledge_base="docs",
        embedder=Embedder(),
        retriever=retriever,
        reranker=Reranker(),
        enable_reranker=True,
    )
    result = asyncio.run(
        service.search(SearchKnowledgeRequest(query="x", limit=2, candidate_limit=3))
    )
    assert result.reranking_status == "applied"
    assert [item.source for item in result.evidence] == ["c", "a"]
    assert [item.reranking_score for item in result.evidence] == [0.9, 0.5]
    assert retriever.limit == 3


def test_reranker_failure_falls_back_to_vector_order_and_no_hits_is_disabled() -> None:
    class Failing:
        def rerank(self, query, candidates):
            raise RuntimeError("provider payload secret")

    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        default_knowledge_base="docs",
        embedder=Embedder(),
        retriever=Retriever(),
        reranker=Failing(),
        enable_reranker=True,
    )
    result = asyncio.run(service.search(SearchKnowledgeRequest(query="x")))
    assert result.outcome == "success"
    assert result.reranking_status == "failed"
    assert [item.source for item in result.evidence] == ["a", "b", "c"]


def test_enabled_empty_retriever_is_no_hits_and_disabled() -> None:
    class Empty(Retriever):
        def retrieve(self, *args, **kwargs):
            return []

    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        default_knowledge_base="docs",
        embedder=Embedder(),
        retriever=Empty(),
        reranker=Reranker(),
        enable_reranker=True,
    )
    result = asyncio.run(service.search(SearchKnowledgeRequest(query="x")))
    assert result.outcome == "no_hits"
    assert result.reranking_status == "disabled"


def test_allowed_false_override_disables_default_reranker() -> None:
    class Counting(Reranker):
        calls = 0

        def rerank(self, query, candidates):
            self.calls += 1
            return super().rerank(query, candidates)

    reranker = Counting()
    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        default_knowledge_base="docs",
        embedder=Embedder(),
        retriever=Retriever(),
        reranker=reranker,
        enable_reranker=True,
        allow_reranker_override=True,
    )
    result = asyncio.run(service.search(SearchKnowledgeRequest(query="x", rerank=False)))
    assert result.reranking_status == "disabled"
    assert reranker.calls == 0


def test_reranker_failure_trace_and_logs_are_secret_free(monkeypatch, caplog) -> None:
    secret = "provider-payload-secret"

    class Span:
        attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    span = Span()

    class Tracer:
        def start_as_current_span(self, name, **kwargs):
            return span

    fake_trace = type(
        "Trace",
        (),
        {"get_tracer": lambda *_: Tracer(), "get_current_span": staticmethod(lambda: span)},
    )()
    monkeypatch.setattr(knowledge_module, "trace", fake_trace)

    class Failing:
        def rerank(self, query, candidates):
            raise RuntimeError(secret)

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(
            OracleKnowledgeService(
                knowledge_bases={"docs": "RAW"},
                default_knowledge_base="docs",
                embedder=Embedder(),
                retriever=Retriever(),
                reranker=Failing(),
                enable_reranker=True,
            ).search(SearchKnowledgeRequest(query="x"))
        )
    assert result.outcome == "success"
    assert span.attributes["oracle.knowledge.error_stage"] == "reranking"
    assert span.attributes["oracle.knowledge.error_type"] == "RuntimeError"
    assert span.attributes["oracle.knowledge.error_code"] == "provider_failure"
    assert secret not in caplog.text
    assert secret not in str(span.attributes)


def test_rerank_override_is_rejected_or_allowed_through_real_mcp_client() -> None:
    async def run():
        service = OracleKnowledgeService(
            knowledge_bases={"docs": "RAW"},
            default_knowledge_base="docs",
            embedder=Embedder(),
            retriever=Retriever(),
            reranker=Reranker(),
            enable_reranker=False,
            allow_reranker_override=True,
        )
        async with Client(create_oracle_knowledge_server(service)) as client:
            result = await client.call_tool("search_knowledge", {"query": "x", "rerank": True})
            assert result.structured_content["reranking_status"] == "applied"

    asyncio.run(run())

import asyncio
from dataclasses import dataclass

from rag_agent.application.oracle_knowledge import OracleKnowledgeService, SearchKnowledgeRequest


@dataclass
class Candidate:
    content: str
    metadata: dict[str, object]
    retrieval_score: float | None = None


class Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class Retriever:
    def retrieve(self, collection, query_embedding, limit, metadata_filters=None):
        return [Candidate("content", {"source": "guide", "page": 3}, 0.4)]

    def list_documents(self, collection):
        return [{"source": "guide", "title": "Guide"}]


def service() -> OracleKnowledgeService:
    return OracleKnowledgeService(
        knowledge_bases={"docs": "RAW_COLLECTION"},
        embedder=Embedder(),
        retriever=Retriever(),
        enable_reranker=False,
        default_knowledge_base="docs",
    )


def test_search_returns_normalized_typed_evidence_without_raw_collection() -> None:
    result = asyncio.run(
        service().search(SearchKnowledgeRequest(query="  find this  ", knowledge_base="docs"))
    )
    assert result.outcome == "success"
    assert result.query == "find this"
    assert result.knowledge_base == "docs"
    assert result.evidence[0].source == "guide"
    assert result.evidence[0].page == "3"
    assert "RAW_COLLECTION" not in result.model_dump_json()


def test_unknown_key_is_forbidden_and_empty_retrieval_is_no_hits() -> None:
    denied = asyncio.run(
        service().search(SearchKnowledgeRequest(query="x", knowledge_base="RAW_COLLECTION"))
    )
    assert denied.outcome == "forbidden"
    assert denied.knowledge_base is None
    assert "RAW_COLLECTION" not in denied.model_dump_json()

    denied_documents = service().list_documents("RAW_COLLECTION")
    assert denied_documents.outcome == "forbidden"
    assert denied_documents.knowledge_base is None
    assert "RAW_COLLECTION" not in denied_documents.model_dump_json()

    class Empty(Retriever):
        def retrieve(self, *args, **kwargs):
            return []

    result_service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        embedder=Embedder(),
        retriever=Empty(),
        enable_reranker=False,
        default_knowledge_base="docs",
    )
    result = asyncio.run(result_service.search(SearchKnowledgeRequest(query="x")))
    assert result.outcome == "no_hits"
    assert result.evidence == []


def test_reranker_override_and_provider_scores() -> None:
    class Reranker:
        def rerank(self, query, candidates):
            return [(candidates[0], 0.91)]

    request = SearchKnowledgeRequest(query="x", rerank=True)
    denied = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        embedder=Embedder(),
        retriever=Retriever(),
        reranker=Reranker(),
        default_knowledge_base="docs",
        enable_reranker=False,
    ).search
    assert asyncio.run(denied(request)).outcome == "invalid_request"
    allowed = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        embedder=Embedder(),
        retriever=Retriever(),
        reranker=Reranker(),
        default_knowledge_base="docs",
        enable_reranker=False,
        allow_reranker_override=True,
    )
    result = asyncio.run(allowed.search(request))
    assert result.reranking_status == "applied"
    assert result.evidence[0].reranking_score == 0.91

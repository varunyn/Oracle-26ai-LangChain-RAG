"""Deterministic production-server fixture for transport contract tests."""

from dataclasses import dataclass

from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from src.rag_agent.application.oracle_knowledge import OracleKnowledgeService


@dataclass
class Candidate:
    content: str
    metadata: dict[str, object]
    retrieval_score: float | None = 0.8


class Embedder:
    def embed_query(self, query):
        if query == "embedding-fail":
            raise RuntimeError("safe embedding failure")
        return [1.0]


class Retriever:
    def retrieve(self, collection, vector, limit, filters=None):
        if collection == "RAW_FAIL":
            raise RuntimeError("safe oracle failure")
        if collection == "RAW_EMPTY":
            return []
        return [
            Candidate("evidence", {"source": "fixture", "document_id": "doc", "chunk_id": "chunk"})
        ]

    def list_documents(self, collection):
        return [{"source": "fixture", "title": "Fixture"}]


class Reranker:
    def rerank(self, query, candidates):
        if query == "rerank-fail":
            raise RuntimeError("safe reranker failure")
        return [(candidates[0], 0.9)]


def create_fixture_server(*, readiness_probe=None):
    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW", "empty": "RAW_EMPTY", "fail": "RAW_FAIL"},
        default_knowledge_base="docs",
        embedder=Embedder(),
        retriever=Retriever(),
        reranker=Reranker(),
        enable_reranker=True,
        allow_reranker_override=True,
    )
    return create_oracle_knowledge_server(service, readiness_probe=readiness_probe)


if __name__ == "__main__":
    create_fixture_server().run(transport="stdio")

from __future__ import annotations

from rag_agent.application.oracle_knowledge import InternalRetrievalResult, RetrievalCandidate
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


class FakeSettings:
    RAG_RETRIEVAL_TOP_K = 3


class FakeService:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def retrieve_candidates(self, query, *, knowledge_base, limit, metadata_filters=None):
        self.calls.append((query, knowledge_base, limit))
        return self.result


def test_oracle_retrieval_tool_returns_error_content_when_vector_search_fails(
    monkeypatch, caplog
) -> None:
    provider_secret = "oracle-wallet-password=do-not-expose"
    service = FakeService(InternalRetrievalResult(outcome="backend_error", error=provider_secret))
    monkeypatch.setattr(
        rag_runtime, "build_oracle_knowledge_service", lambda *args, **kwargs: service
    )

    evidence = OracleRetrievalEvidenceStore()
    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="ORACLE_WEB_EMBEDDINGS",
        filter_docs=lambda query, docs: docs,
        evidence=evidence,
    )

    content = tool.invoke(
        {
            "type": "tool_call",
            "id": "oracle-call-1",
            "name": "oracle_retrieval",
            "args": {"query": "Northway Solutions payment terms"},
        }
    )

    assert "Oracle retrieval failed" in content.content
    assert "knowledge backend unavailable" in content.content
    assert provider_secret not in content.content
    assert provider_secret not in str(content.artifact)
    assert provider_secret not in caplog.text
    selected = evidence.read()
    assert selected is not None
    assert selected.invocation_id == "oracle-call-1"
    assert selected.collection_name == "ORACLE_WEB_EMBEDDINGS"
    assert selected.query == "Northway Solutions payment terms"
    assert selected.documents == []
    assert selected.error == "knowledge backend unavailable"


def test_oracle_retrieval_tool_uses_configured_top_k(monkeypatch) -> None:
    service = FakeService(
        InternalRetrievalResult(
            outcome="success",
            candidates=[
                RetrievalCandidate(
                    "Payment terms are Net 30.",
                    {"source": "guide", "document_id": "d1", "chunk_id": "c1"},
                )
            ],
        )
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        rag_runtime, "build_oracle_knowledge_service", lambda *args, **kwargs: service
    )

    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="ORACLE_WEB_EMBEDDINGS",
        filter_docs=lambda query, docs: docs,
        evidence=OracleRetrievalEvidenceStore(),
    )

    content = tool.invoke(
        {
            "type": "tool_call",
            "id": "oracle-call-2",
            "name": "oracle_retrieval",
            "args": {"query": "Northway Solutions payment terms"},
        }
    )

    assert "Payment terms are Net 30" in content.content
    assert service.calls[0][2] == 3

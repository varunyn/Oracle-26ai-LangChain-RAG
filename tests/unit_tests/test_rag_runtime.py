from __future__ import annotations

from types import TracebackType

from langchain_core.documents import Document

from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


class FakeSettings:
    RAG_RETRIEVAL_TOP_K = 3


class FakeConnectionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def test_oracle_retrieval_tool_returns_error_content_when_vector_search_fails(
    monkeypatch,
) -> None:
    def failing_search_documents(**kwargs: object) -> list[object]:
        _ = kwargs
        raise RuntimeError("Failed due to a DB error: ORA-22275: invalid LOB locator specified")

    monkeypatch.setattr(rag_runtime, "get_pooled_connection", lambda: FakeConnectionContext())
    monkeypatch.setattr(rag_runtime, "get_embedding_model", lambda: object())
    monkeypatch.setattr(rag_runtime, "search_documents", failing_search_documents)

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
    assert "ORA-22275" in content.content
    selected = evidence.read()
    assert selected is not None
    assert selected.invocation_id == "oracle-call-1"
    assert selected.collection_name == "ORACLE_WEB_EMBEDDINGS"
    assert selected.query == "Northway Solutions payment terms"
    assert selected.documents == []
    assert selected.error == "Failed due to a DB error: ORA-22275: invalid LOB locator specified"


def test_oracle_retrieval_tool_uses_configured_top_k(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_search_documents(**kwargs: object) -> list[Document]:
        captured.update(kwargs)
        return [Document(page_content="Payment terms are Net 30.")]

    monkeypatch.setattr("api.settings.get_settings", lambda: FakeSettings())
    monkeypatch.setattr(rag_runtime, "get_pooled_connection", lambda: FakeConnectionContext())
    monkeypatch.setattr(rag_runtime, "get_embedding_model", lambda: object())
    monkeypatch.setattr(rag_runtime, "search_documents", fake_search_documents)

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
    assert captured["top_k"] == 3

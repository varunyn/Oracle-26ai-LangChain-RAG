from langchain_core.documents import Document

from src.rag_agent.application.oracle_knowledge import InternalRetrievalResult, RetrievalCandidate
from src.rag_agent.infrastructure import oracle_knowledge as infrastructure
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


class Service:
    def __init__(self):
        self.calls = []
        self.candidates = [
            RetrievalCandidate(
                "chunk",
                {
                    "source": "guide",
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "custom": "preserve",
                },
            )
        ]

    async def retrieve_candidates(self, query, *, knowledge_base, limit, metadata_filters=None):
        self.calls.append(("retrieve", knowledge_base, limit))
        return InternalRetrievalResult(outcome="success", candidates=self.candidates)

    async def rerank_candidates(self, query, candidates, *, enabled):
        self.calls.append(("rerank", enabled))
        return ([(candidate, 0.9) for candidate in candidates], "applied")


def test_all_chat_helpers_delegate_to_shared_service_and_preserve_metadata(monkeypatch):
    service = Service()
    monkeypatch.setattr(
        rag_runtime, "build_oracle_knowledge_service", lambda *args, **kwargs: service
    )
    docs = rag_runtime.retrieve_oracle_docs(query="q", collection_name="RAW_CHAT_COLLECTION", k=4)
    assert docs[0].metadata["custom"] == "preserve"
    assert service.calls[0] == ("retrieve", "chat", 4)

    reranked = rag_runtime.rerank_retrieved_docs("q", docs, enable_reranker=True)
    assert reranked[0].metadata["chunk_id"] == "chunk-1"
    assert any(call[0] == "rerank" for call in service.calls)

    evidence = OracleRetrievalEvidenceStore()
    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="RAW_CHAT_COLLECTION", filter_docs=lambda q, d: d, evidence=evidence
    )
    result = tool.invoke(
        {"type": "tool_call", "id": "call-1", "name": "oracle_retrieval", "args": {"query": "q"}}
    )
    assert "chunk" in result.content
    assert evidence.read().invocation_id == "call-1"


def test_backend_error_and_no_hits_are_distinct(monkeypatch):
    class Failing(Service):
        async def retrieve_candidates(self, *args, **kwargs):
            return InternalRetrievalResult(
                outcome="backend_error", error="provider-password-do-not-expose"
            )

    service = Failing()
    monkeypatch.setattr(
        rag_runtime, "build_oracle_knowledge_service", lambda *args, **kwargs: service
    )
    try:
        rag_runtime.retrieve_oracle_docs(query="q", collection_name="RAW", k=1)
    except RuntimeError as exc:
        assert str(exc) == "knowledge backend unavailable"
        assert "provider-password-do-not-expose" not in str(exc)
    else:
        raise AssertionError("backend error must not become no hits")


def test_reranker_provider_error_is_not_logged(monkeypatch, caplog):
    provider_secret = "reranker-token-do-not-expose"

    class FailingReranker(Service):
        async def rerank_candidates(self, *args, **kwargs):
            raise RuntimeError(provider_secret)

    monkeypatch.setattr(
        rag_runtime,
        "build_oracle_knowledge_service",
        lambda *args, **kwargs: FailingReranker(),
    )
    docs = [Document(page_content="Oracle CLI configuration", metadata={"source": "guide"})]

    result = rag_runtime.rerank_retrieved_docs(
        "Oracle CLI configuration", docs, enable_reranker=True
    )

    assert result == docs
    assert "oci_rerank_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert provider_secret not in caplog.text


def test_shared_factory_constructs_real_adapter_protocols(monkeypatch):
    monkeypatch.setattr(infrastructure, "get_embedding_model", lambda *_: object())
    settings = type(
        "Settings",
        (),
        {
            "DEFAULT_COLLECTION": "RAW",
            "ORACLE_KNOWLEDGE_ENABLE_RERANKER": False,
            "EMBED_MODEL_TYPE": "OCI",
        },
    )()
    service = infrastructure.build_oracle_knowledge_service(settings)
    adapter = service._retriever
    assert (
        hasattr(adapter, "retrieve")
        and hasattr(adapter, "list_documents")
        and hasattr(adapter, "rerank")
    )


def test_factory_empty_allowlist_fails_closed(monkeypatch):
    monkeypatch.setattr(infrastructure, "get_embedding_model", lambda *_: object())
    settings = type(
        "Settings",
        (),
        {
            "DEFAULT_COLLECTION": "RAW",
            "ORACLE_KNOWLEDGE_ENABLE_RERANKER": False,
            "EMBED_MODEL_TYPE": "OCI",
        },
    )()
    try:
        infrastructure.build_oracle_knowledge_service(settings, knowledge_bases={})
    except ValueError as exc:
        assert "default knowledge base" in str(exc)
    else:
        raise AssertionError("empty MCP allowlist must fail closed")


def test_factory_chat_collection_uses_internal_chat_key(monkeypatch):
    monkeypatch.setattr(infrastructure, "get_embedding_model", lambda *_: object())
    settings = type(
        "Settings",
        (),
        {
            "DEFAULT_COLLECTION": "OTHER",
            "ORACLE_KNOWLEDGE_ENABLE_RERANKER": False,
            "EMBED_MODEL_TYPE": "OCI",
        },
    )()
    service = infrastructure.build_oracle_knowledge_service(settings, collection_name="RAW_CHAT")
    assert service._knowledge_bases == {"chat": "RAW_CHAT"}
    assert service._enable_reranker is False

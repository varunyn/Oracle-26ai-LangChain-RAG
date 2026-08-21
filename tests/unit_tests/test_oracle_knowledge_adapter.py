from contextlib import nullcontext

from langchain_core.documents import Document

from rag_agent.infrastructure import oracle_knowledge


class Store:
    def __init__(self):
        self.calls = []

    def similarity_search_by_vector_with_relevance_scores(self, embedding, **kwargs):
        self.calls.append((embedding, kwargs))
        return [(Document(page_content="chunk", metadata={"source": "guide"}), 0.73)]


def test_oracle_adapter_propagates_retrieval_score(monkeypatch) -> None:
    adapter = oracle_knowledge.OracleKnowledgeAdapter.__new__(
        oracle_knowledge.OracleKnowledgeAdapter
    )
    adapter._embedder = object()
    monkeypatch.setattr(oracle_knowledge, "get_connection", lambda: nullcontext(object()))
    store = Store()
    monkeypatch.setattr(oracle_knowledge, "get_oracle_vs", lambda *args, **kwargs: store)
    candidates = adapter.retrieve("RAW_COLLECTION", [1.0], 1)
    assert candidates[0].retrieval_score == 0.73
    assert store.calls == [([1.0], {"k": 1})]


def test_oracle_adapter_sends_only_nonempty_metadata_filters(monkeypatch) -> None:
    adapter = oracle_knowledge.OracleKnowledgeAdapter.__new__(
        oracle_knowledge.OracleKnowledgeAdapter
    )
    adapter._embedder = object()
    store = Store()
    monkeypatch.setattr(oracle_knowledge, "get_connection", lambda: nullcontext(object()))
    monkeypatch.setattr(oracle_knowledge, "get_oracle_vs", lambda *args, **kwargs: store)

    adapter.retrieve("RAW_COLLECTION", [1.0], 2, {"source": "guide"})

    assert store.calls == [([1.0], {"k": 2, "filter": {"source": "guide"}})]

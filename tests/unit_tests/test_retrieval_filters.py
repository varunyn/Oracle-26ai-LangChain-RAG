from langchain_core.documents import Document

from src.rag_agent.infrastructure.retrieval import (
    _apply_metadata_filters,
    normalize_collection_name,
    normalize_search_mode,
    search_documents,
)


class FakeVectorStore:
    last_collection_name: str | None = None
    last_query: str | None = None
    last_k: int | None = None
    last_filter: dict[str, object] | None = None

    def similarity_search(self, *, query: str, k: int, filter=None):
        type(self).last_query = query
        type(self).last_k = k
        type(self).last_filter = filter
        return [
            Document(page_content="alpha", metadata={"language": "en"}),
            Document(page_content="beta", metadata={"language": "fr"}),
        ]


def test_apply_metadata_filters_keeps_matching_docs() -> None:
    docs = [
        Document(page_content="alpha", metadata={"language": "en", "product_area": "cli"}),
        Document(page_content="beta", metadata={"language": "fr", "product_area": "cli"}),
    ]

    filtered = _apply_metadata_filters(docs, {"language": "en"})

    assert [doc.page_content for doc in filtered] == ["alpha"]


def test_apply_metadata_filters_returns_all_docs_when_filters_empty() -> None:
    docs = [Document(page_content="alpha", metadata={"language": "en"})]

    filtered = _apply_metadata_filters(docs, None)

    assert [doc.page_content for doc in filtered] == ["alpha"]


def test_normalize_search_mode_only_allows_vector() -> None:
    assert normalize_search_mode("vector") == "vector"
    assert normalize_search_mode("hybrid") == "vector"
    assert normalize_search_mode("text") == "vector"


def test_normalize_collection_name_uppercases_simple_oracle_identifiers() -> None:
    assert normalize_collection_name("oracle_web_embeddings") == "ORACLE_WEB_EMBEDDINGS"
    assert normalize_collection_name(" Rag_Knowledge_Base ") == "RAG_KNOWLEDGE_BASE"


def test_normalize_collection_name_preserves_explicit_or_special_identifiers() -> None:
    assert normalize_collection_name('"oracle_web_embeddings"') == '"oracle_web_embeddings"'
    assert normalize_collection_name("schema.collection") == "schema.collection"


def test_search_documents_passes_metadata_filters_to_oraclevs(monkeypatch) -> None:
    fake_store = FakeVectorStore()

    def fake_get_oracle_vs(conn, collection_name, embed_model):
        _ = conn, embed_model
        FakeVectorStore.last_collection_name = collection_name
        return fake_store

    monkeypatch.setattr(
        "src.rag_agent.infrastructure.retrieval.get_oracle_vs",
        fake_get_oracle_vs,
    )

    docs = search_documents(
        conn=object(),
        collection_name="coll_a",
        embed_model=object(),
        query="hello",
        top_k=3,
        search_mode="vector",
        metadata_filters={"language": "en"},
    )

    assert [doc.page_content for doc in docs] == ["alpha", "beta"]
    assert FakeVectorStore.last_collection_name == "COLL_A"
    assert FakeVectorStore.last_query == "hello"
    assert FakeVectorStore.last_k == 3
    assert FakeVectorStore.last_filter == {"language": "en"}

from langchain_core.documents import Document

from src.rag_agent.infrastructure.retrieval import _apply_metadata_filters


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

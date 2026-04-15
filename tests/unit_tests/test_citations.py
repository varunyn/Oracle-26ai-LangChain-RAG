from langchain_core.documents import Document

from src.rag_agent.core.citations import citations_from_documents, normalize_citations


def test_normalize_citations_handles_mixed_metadata_keys() -> None:
    raw = [
        {
            "link": "https://docs.oracle.com/vb",
            "title": "Visual Applications",
            "page_number": 7,
        },
        {
            "url": "https://docs.oracle.com/apex",
            "document_name": "APEX App Builder Guide",
        },
        {
            "source": "direct-source",
            "page": 3,
            "source_url": "https://example.com/direct-source",
        },
    ]

    assert normalize_citations(raw) == [
        {
            "source": "Visual Applications",
            "page": "7",
            "link": "https://docs.oracle.com/vb",
        },
        {
            "source": "APEX App Builder Guide",
            "page": None,
            "link": "https://docs.oracle.com/apex",
        },
        {
            "source": "direct-source",
            "page": "3",
            "link": "https://example.com/direct-source",
        },
    ]


def test_citations_from_documents_uses_normalized_contract() -> None:
    docs = [
        Document(
            page_content="A",
            metadata={
                "file_name": "guide.md",
                "source_url": "https://docs.oracle.com/guide",
            },
        )
    ]

    assert citations_from_documents(docs) == [
        {
            "source": "guide.md",
            "page": None,
            "link": "https://docs.oracle.com/guide",
        }
    ]


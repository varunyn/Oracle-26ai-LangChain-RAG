from __future__ import annotations

from langchain_core.documents import Document

from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


def test_retrieval_evidence_reads_the_latest_completed_oracle_lookup() -> None:
    evidence = OracleRetrievalEvidenceStore()
    first_document = Document(page_content="First result", metadata={"source": "first.md"})

    evidence.record(
        invocation_id="oracle-call-1",
        query="first question",
        documents=[first_document],
    )
    evidence.record(
        invocation_id="oracle-call-2",
        query="second question",
        documents=[],
        error="Oracle connection timed out",
    )

    selected = evidence.read()

    assert selected is not None
    assert selected.invocation_id == "oracle-call-2"
    assert selected.query == "second question"
    assert selected.documents == []
    assert selected.error == "Oracle connection timed out"


def test_retrieval_evidence_distinguishes_empty_results_from_failures() -> None:
    evidence = OracleRetrievalEvidenceStore()

    evidence.record(
        invocation_id="oracle-call-1",
        query="missing policy",
        documents=[],
    )

    selected = evidence.read()

    assert selected is not None
    assert selected.invocation_id == "oracle-call-1"
    assert selected.documents == []
    assert selected.error is None

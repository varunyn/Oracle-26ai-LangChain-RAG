"""Turn-scoped evidence exchanged by Oracle retrieval and mixed-mode composition."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class OracleRetrievalEvidence:
    """The completed result of one Oracle collection retrieval invocation."""

    invocation_id: str | None
    query: str
    documents: list[Document]
    error: str | None
    collection_name: str


class OracleRetrievalEvidenceStore:
    """Record and select the latest completed Oracle retrieval for one turn."""

    def __init__(self) -> None:
        self._latest: OracleRetrievalEvidence | None = None

    def record(
        self,
        *,
        invocation_id: str | None,
        query: str,
        documents: list[Document],
        error: str | None = None,
        collection_name: str = "RAG_KNOWLEDGE_BASE",
    ) -> OracleRetrievalEvidence:
        """Record a completed retrieval; the latest completed invocation is selected."""
        evidence = OracleRetrievalEvidence(
            invocation_id=invocation_id,
            query=query,
            documents=list(documents),
            error=error,
            collection_name=collection_name,
        )
        self._latest = evidence
        return evidence

    def read(self) -> OracleRetrievalEvidence | None:
        """Return the evidence selected for this turn, if Oracle retrieval completed."""
        return self._latest

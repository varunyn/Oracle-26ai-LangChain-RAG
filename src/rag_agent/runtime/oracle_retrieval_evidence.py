"""Turn-scoped evidence exchanged by Oracle retrieval and mixed-mode composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall


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

    @classmethod
    def from_persisted_messages(
        cls, messages: Sequence[object], *, collection_name: str
    ) -> OracleRetrievalEvidenceStore:
        """Rebuild current-turn evidence from the persisted tool artifact."""
        store = cls()
        calls: dict[str, ToolCall] = {}
        latest_human = max(
            (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
            default=-1,
        )
        for message in messages[latest_human + 1 :]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    if str(call.get("name") or "") == "oracle_retrieval":
                        calls[str(call.get("id") or "")] = call
                continue
            if not isinstance(message, ToolMessage) or message.name != "oracle_retrieval":
                continue
            matched_call = calls.get(str(message.tool_call_id))
            args = matched_call["args"] if matched_call is not None else {}
            query = args.get("query", args.get("q", ""))
            artifact_error = next(
                (
                    str(item["error"])
                    for item in (message.artifact or [])
                    if isinstance(item, dict)
                    and item.get("type") == "oracle_retrieval_error"
                    and item.get("error")
                ),
                None,
            )
            documents = [
                document
                for item in (message.artifact or [])
                if (document := _document_from_artifact(item)) is not None
            ]
            error = artifact_error or (
                str(message.content or "") if message.status == "error" else None
            )
            store.record(
                invocation_id=str(message.tool_call_id),
                query=str(query),
                documents=documents,
                error=error,
                collection_name=collection_name,
            )
        return store


def _document_from_artifact(value: object) -> Document | None:
    if isinstance(value, Document):
        return value
    if not isinstance(value, dict):
        return None
    payload = value.get("data") if value.get("type") == "Document" and "data" in value else value
    if not isinstance(payload, dict) or "page_content" not in payload:
        return None
    metadata = payload.get("metadata")
    return Document(
        page_content=str(payload["page_content"]),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )

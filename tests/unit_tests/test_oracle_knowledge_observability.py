from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from fastmcp import Client
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import src.rag_agent.application.oracle_knowledge as knowledge_module
from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from src.rag_agent.application.oracle_knowledge import OracleKnowledgeService
from src.rag_agent.utils.logging_config import REQUEST_ID_CTX

MARKERS = {
    "sentinel-query-unique",
    "private-document-unique",
    "private-metadata-unique",
    "password-unique",
    "token-unique",
    "wallet/config-unique",
    "dsn.internal:1521/ORCL",
    "SELECT secret_unique",
    "RAW_TABLE_UNIQUE",
    "raw-provider-exception-unique",
    "provider-payload-unique",
}


@dataclass
class Candidate:
    content: str
    metadata: dict[str, object]
    retrieval_score: float | None = 0.7


class Embedder:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def embed_query(self, query):
        if self.error:
            raise self.error
        return [1.0]


class Retriever:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def retrieve(self, collection, vector, limit, filters=None):
        if self.error:
            raise self.error
        return [
            Candidate(
                "private-document-unique",
                {"source": "private-metadata-unique", "raw": "RAW_TABLE_UNIQUE"},
            )
        ]

    def list_documents(self, collection):
        return []


class Reranker:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def rerank(self, query, candidates):
        if self.error:
            raise self.error
        return [(candidates[0], 0.91)]


def _service(embedder=None, retriever=None, reranker=None):
    return OracleKnowledgeService(
        knowledge_bases={"docs": "RAW_TABLE_UNIQUE"},
        default_knowledge_base="docs",
        embedder=embedder or Embedder(),
        retriever=retriever or Retriever(),
        reranker=reranker or Reranker(),
        enable_reranker=True,
    )


def _span_text(spans) -> str:
    parts = []
    for span in spans:
        parts.extend([span.name, str(span.attributes), str(span.status.description)])
        for event in span.events:
            parts.extend([event.name, str(event.attributes)])
    return " ".join(parts)


def _patch_tracers(monkeypatch, provider):
    tracer = provider.get_tracer("oracle-knowledge-test")
    monkeypatch.setattr(
        knowledge_module,
        "trace",
        SimpleNamespace(get_tracer=lambda *_: tracer, get_current_span=otel_trace.get_current_span),
    )


def test_fastmcp_observability_spans_request_id_and_secret_safety(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _patch_tracers(monkeypatch, provider)
    sentinel = REQUEST_ID_CTX.set("prior-sentinel")
    try:

        async def run():
            async with Client(create_oracle_knowledge_server(_service())) as client:
                result = await client.call_tool(
                    "search_knowledge",
                    {"query": "sentinel-query-unique", "knowledge_base": "docs", "limit": 1},
                )
                assert result.structured_content["outcome"] == "success"

        asyncio.run(run())
        spans = exporter.get_finished_spans()
        names = {span.name for span in spans}
        assert {
            "oracle.knowledge.search",
            "oracle.knowledge.embedding",
            "oracle.knowledge.search_oracle",
            "oracle.knowledge.reranking",
        } <= names
        request_ids = [
            span.attributes.get("oracle.knowledge.request_id")
            for span in spans
            if span.name.startswith("oracle.knowledge")
        ]
        assert request_ids and all(value not in (None, "-") for value in request_ids)
        text = _span_text(spans)
        assert not any(marker in text for marker in MARKERS)
        embedding = next(span for span in spans if span.name == "oracle.knowledge.embedding")
        assert embedding.attributes["oracle.knowledge.query_length"] == len("sentinel-query-unique")
        assert REQUEST_ID_CTX.get() == "prior-sentinel"
    finally:
        REQUEST_ID_CTX.reset(sentinel)
    assert REQUEST_ID_CTX.get() == "-"


def test_failure_spans_only_contain_safe_stage_type_code(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _patch_tracers(monkeypatch, provider)

    async def run(service, expected_outcome="backend_error"):
        async with Client(create_oracle_knowledge_server(service)) as client:
            result = await client.call_tool(
                "search_knowledge", {"query": "sentinel-query-unique", "knowledge_base": "docs"}
            )
            assert result.structured_content["outcome"] == expected_outcome

    asyncio.run(
        run(
            _service(
                embedder=Embedder(
                    RuntimeError("raw-provider-exception-unique provider-payload-unique")
                )
            )
        )
    )
    spans = exporter.get_finished_spans()
    text = _span_text(spans)
    assert not any(marker in text for marker in MARKERS)
    total = next(span for span in spans if span.name == "oracle.knowledge.search")
    assert total.attributes["oracle.knowledge.outcome"] == "backend_error"
    assert any(span.attributes.get("oracle.knowledge.error_stage") == "embedding" for span in spans)
    exporter.clear()
    asyncio.run(
        run(
            _service(
                retriever=Retriever(
                    RuntimeError("retrieval-provider-payload-unique password-unique")
                )
            )
        )
    )
    spans = exporter.get_finished_spans()
    assert any(span.attributes.get("oracle.knowledge.error_stage") == "retrieval" for span in spans)
    exporter.clear()
    asyncio.run(
        run(
            _service(
                reranker=Reranker(RuntimeError("rerank-provider-payload-unique token-unique"))
            ),
            "success",
        )
    )
    spans = exporter.get_finished_spans()
    assert any(span.attributes.get("oracle.knowledge.error_stage") == "reranking" for span in spans)
    assert not any(marker in _span_text(spans) for marker in MARKERS)

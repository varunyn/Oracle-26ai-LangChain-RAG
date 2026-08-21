import asyncio
import logging
import time

from fastmcp import Client

import rag_agent.application.oracle_knowledge as knowledge_module
from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from rag_agent.application.oracle_knowledge import OracleKnowledgeService, SearchKnowledgeRequest


class Embedder:
    def __init__(self, error: Exception | None = None, delay: float = 0) -> None:
        self.error, self.delay, self.calls = error, delay, 0

    async def embed_query(self, query: str):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return [1.0]


class Retriever:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.calls = error, 0

    async def retrieve(self, *args):
        self.calls += 1
        if self.error:
            raise self.error
        return []

    async def list_documents(self, collection):
        return []


def make(
    *, embedder=None, retriever=None, timeout=1.0, query=8, results=5, candidates=5, filters=2
):
    return OracleKnowledgeService(
        knowledge_bases={"docs": "RAW_TABLE"},
        default_knowledge_base="docs",
        embedder=embedder or Embedder(),
        retriever=retriever or Retriever(),
        enable_reranker=False,
        max_query_length=query,
        max_result_limit=results,
        max_candidate_limit=candidates,
        max_metadata_filters=filters,
        execution_timeout_seconds=timeout,
    )


def test_timeout_cancels_async_provider_and_prevents_retrieval() -> None:
    embedder, retriever = Embedder(delay=0.2), Retriever()
    result = asyncio.run(
        make(embedder=embedder, retriever=retriever, timeout=0.01).search(
            SearchKnowledgeRequest(query="ok")
        )
    )
    assert result.outcome == "backend_error"
    assert retriever.calls == 0


def test_sync_provider_is_offloaded_and_timeout_returns_on_time() -> None:
    class Slow:
        def embed_query(self, query):
            import time

            time.sleep(0.2)
            return [1.0]

    loop = asyncio.new_event_loop()
    started = time.perf_counter()
    try:
        result = loop.run_until_complete(
            make(embedder=Slow(), timeout=0.01).search(SearchKnowledgeRequest(query="ok"))
        )
    finally:
        loop.close()
    assert time.perf_counter() - started < 0.15
    assert result.outcome == "backend_error"


def test_sync_timeout_worker_finishes_without_downstream_work() -> None:
    class Slow:
        def embed_query(self, query):
            time.sleep(0.05)
            return [1.0]

    retriever = Retriever()
    result = asyncio.run(
        make(embedder=Slow(), retriever=retriever, timeout=0.01).search(
            SearchKnowledgeRequest(query="ok")
        )
    )
    assert result.outcome == "backend_error"
    time.sleep(0.08)
    assert retriever.calls == 0


def test_forbidden_key_is_checked_before_providers() -> None:
    embedder, retriever = Embedder(), Retriever()
    result = asyncio.run(
        make(embedder=embedder, retriever=retriever).search(
            SearchKnowledgeRequest(query="ok", knowledge_base="RAW_TABLE")
        )
    )
    assert result.outcome == "forbidden"
    assert embedder.calls == retriever.calls == 0


def test_provider_failures_are_sanitized_in_result_and_logs(caplog) -> None:
    secret = "password=PW123 token=TK456 wallet=/private/wallet dsn=db.internal SQL=SELECT secret provider_payload={raw-body-marker} Oracle connection dbhost:1521/ORCL"

    class SyncFail:
        calls = 0

        def embed_query(self, query):
            self.calls += 1
            raise RuntimeError(secret)

    embedder = SyncFail()
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(make(embedder=embedder).search(SearchKnowledgeRequest(query="ok")))
    text = result.model_dump_json() + " " + caplog.text
    assert result.outcome == "backend_error"
    assert result.error == "knowledge backend unavailable"
    for value in (
        "PW123",
        "TK456",
        "/private/wallet",
        "db.internal",
        "SELECT secret",
        "{raw-body-marker}",
        "dbhost:1521/ORCL",
    ):
        assert value not in text


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class RecordingTracer:
    def __init__(self):
        self.span = RecordingSpan()

    def start_as_current_span(self, name, **kwargs):
        return self.span

    def get_current_span(self):
        return self.span


def test_failure_trace_attributes_are_actionable_and_secret_free(monkeypatch) -> None:
    secret = "PW123 provider_payload={raw-body-marker} dbhost:1521/ORCL"
    tracer = RecordingTracer()
    trace_fake = type(
        "Trace",
        (),
        {"get_tracer": lambda *_: tracer, "get_current_span": staticmethod(lambda: tracer.span)},
    )()
    monkeypatch.setattr(knowledge_module, "trace", trace_fake)

    class FailingEmbedder:
        def embed_query(self, query):
            raise RuntimeError(secret)

    result = asyncio.run(
        make(embedder=FailingEmbedder()).search(SearchKnowledgeRequest(query="ok"))
    )
    assert result.outcome == "backend_error"
    assert tracer.span.attributes["oracle.knowledge.error_stage"] == "embedding"
    assert tracer.span.attributes["oracle.knowledge.error_type"] == "RuntimeError"
    assert tracer.span.attributes["oracle.knowledge.error_code"] == "provider_failure"

    class FailingRetriever(Retriever):
        async def retrieve(self, *args):
            raise OSError(secret)

    tracer.span = RecordingSpan()
    result = asyncio.run(
        make(retriever=FailingRetriever()).search(SearchKnowledgeRequest(query="ok"))
    )
    assert result.outcome == "backend_error"
    assert tracer.span.attributes["oracle.knowledge.error_stage"] == "retrieval"
    assert tracer.span.attributes["oracle.knowledge.error_type"] == "OSError"

    tracer.span = RecordingSpan()
    result = asyncio.run(
        make(embedder=Embedder(delay=0.2), timeout=0.01).search(SearchKnowledgeRequest(query="ok"))
    )
    assert result.outcome == "backend_error"
    assert tracer.span.attributes["oracle.knowledge.error_stage"] == "total"
    assert tracer.span.attributes["oracle.knowledge.error_type"] == "TimeoutError"
    assert tracer.span.attributes["oracle.knowledge.error_code"] == "timeout"
    assert all(secret not in str(value) for value in tracer.span.attributes.values())


def test_mcp_boundaries_and_failures() -> None:
    async def run():
        embedder, retriever = Embedder(), Retriever()
        async with Client(
            create_oracle_knowledge_server(make(embedder=embedder, retriever=retriever))
        ) as client:
            for args in (
                {"query": "   "},
                {"query": "123456789"},
                {"query": "ok", "limit": 6},
                {"query": "ok", "candidate_limit": 6},
                {"query": "ok", "metadata_filters": {"source": "a", "title": "b", "page": 1}},
                {"query": "ok", "metadata_filters": {"raw": "x"}},
                {"query": "ok", "knowledge_base": "RAW_TABLE"},
            ):
                result = await client.call_tool("search_knowledge", args)
                assert result.structured_content["outcome"] in {"invalid_request", "forbidden"}
            assert embedder.calls == retriever.calls == 0

    asyncio.run(run())

import os
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.testclient import TestClient

from src.rag_agent.utils.otel_tracing import (
    _create_resource,
    setup_otel_tracing,
    setup_otel_tracing_early,
)


@pytest.fixture(autouse=True)
def isolate_otel_tracing(monkeypatch: pytest.MonkeyPatch):
    """Keep module setup state and instrumentation isolated across test orderings."""
    import src.rag_agent.utils.otel_tracing as otel_tracing_mod

    monkeypatch.setattr(otel_tracing_mod, "_INITIALIZED", False)
    monkeypatch.setattr(otel_tracing_mod, "_EARLY_PROVIDER", None)
    monkeypatch.setattr(otel_tracing_mod, "BatchSpanProcessor", SimpleSpanProcessor)
    FastAPIInstrumentor().uninstrument()
    RequestsInstrumentor().uninstrument()
    yield
    FastAPIInstrumentor().uninstrument()
    RequestsInstrumentor().uninstrument()


@contextmanager
def temp_env(**env):
    old = {}
    try:
        for k, v in env.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():  # pragma: no cover - trivial
        return {"status": "ok"}

    return app


def test_resource_includes_langfuse_environment():
    with temp_env(LANGFUSE_TRACING_ENVIRONMENT="staging"):
        resource = _create_resource("rag-api")

    assert resource.attributes["service.name"] == "rag-api"
    assert resource.attributes["deployment.environment.name"] == "staging"


def test_early_setup_configures_langsmith_otel_mode(monkeypatch):
    import src.rag_agent.utils.otel_tracing as otel_tracing_mod

    monkeypatch.setattr(otel_tracing_mod, "_EARLY_PROVIDER", None)
    # Keep this environment-contract test from installing a global provider for
    # the request instrumentation test that follows it.
    monkeypatch.setattr(otel_tracing_mod.trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setenv("ENABLE_OTEL_TRACING", "1")
    monkeypatch.delenv("LANGSMITH_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("LANGSMITH_OTEL_ONLY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_MODE", raising=False)

    assert setup_otel_tracing_early() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_TRACING_MODE"] == "otel"
    assert "LANGSMITH_OTEL_ENABLED" not in os.environ
    assert "LANGSMITH_OTEL_ONLY" not in os.environ


def test_fastapi_request_emits_spans_and_service_name():
    exporter = InMemorySpanExporter()
    app = make_app()

    # Enable OTel with in-memory exporter
    initialized = setup_otel_tracing(app=app, exporter=exporter)
    assert initialized is True

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200

    # Ensure async processor flushes before assertions
    from opentelemetry import trace as _trace

    _trace.get_tracer_provider().force_flush()  # type: ignore[attr-defined]

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1

    # Verify Resource service.name is set on provider/spans
    # At least one span should carry the service.name resource attribute
    assert any(
        s.resource.attributes.get("service.name") == "rag-api"  # type: ignore[attr-defined]
        for s in spans
    )

    # Verify we captured the request to /health
    assert any(
        (s.attributes.get("http.route") == "/health") or (s.name.endswith("/health")) for s in spans
    )


def test_fail_open_when_collector_unreachable():
    # Reset so this test actually runs setup with the unreachable endpoint
    import src.rag_agent.utils.otel_tracing as otel_tracing_mod

    otel_tracing_mod._INITIALIZED = False  # type: ignore[attr-defined]

    app = make_app()
    with temp_env(
        ENABLE_OTEL_TRACING="1", OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://127.0.0.1:9/v1/traces"
    ):
        # Should not raise even if exporter cannot reach collector
        setup_otel_tracing(app=app)
        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200

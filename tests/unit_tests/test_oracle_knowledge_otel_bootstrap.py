from pathlib import Path
from types import SimpleNamespace

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mcp_servers.oracle_knowledge as server_module
import src.rag_agent.utils.otel_tracing as otel_module


def test_shared_env_template_does_not_relabel_product_otel_service():
    env_text = (Path(__file__).parents[2] / ".env.example").read_text()
    active_lines = [
        line.strip()
        for line in env_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "OTEL_SERVICE_NAME=oracle-knowledge-mcp" not in active_lines
    assert otel_module.setup_otel_tracing.__defaults__[-1] == "rag-api"


def test_explicit_otel_service_name_is_used_without_global_provider_mutation(monkeypatch):
    captured = {}
    monkeypatch.setattr(otel_module, "_env_enabled", lambda *args, **kwargs: True)
    monkeypatch.setattr(otel_module, "_EARLY_PROVIDER", None)
    monkeypatch.setattr(otel_module, "OTLPSpanExporter", object())
    monkeypatch.setattr(
        otel_module.trace,
        "set_tracer_provider",
        lambda provider: captured.setdefault("provider", provider),
    )

    assert otel_module.setup_otel_tracing_early(service_name="oracle-knowledge-mcp") is True
    assert captured["provider"].resource.attributes["service.name"] == "oracle-knowledge-mcp"


def test_complete_setup_accepts_explicit_service_name(monkeypatch):
    captured = {}
    monkeypatch.setattr(otel_module, "_INITIALIZED", False)
    monkeypatch.setattr(otel_module, "_EARLY_PROVIDER", None)
    monkeypatch.setattr(otel_module, "_env_enabled", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        otel_module.trace,
        "set_tracer_provider",
        lambda provider: captured.setdefault("provider", provider),
    )
    monkeypatch.setattr(FastAPIInstrumentor, "instrument", lambda *args, **kwargs: None)
    monkeypatch.setattr(RequestsInstrumentor, "instrument", lambda *args, **kwargs: None)

    assert (
        otel_module.setup_otel_tracing(
            exporter=InMemorySpanExporter(), service_name="oracle-knowledge-mcp"
        )
        is True
    )
    assert captured["provider"].resource.attributes["service.name"] == "oracle-knowledge-mcp"


def test_standalone_main_bootstraps_logging_and_tracing_before_clients(monkeypatch):
    events = []
    settings = SimpleNamespace(
        ORACLE_KNOWLEDGE_TRANSPORT="stdio", ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING=False
    )

    class FakeServer:
        def run(self, **kwargs):
            events.append(("run", kwargs))

    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        server_module,
        "setup_logging",
        lambda: events.append(("logging", __import__("os").environ.get("OTEL_SERVICE_NAME"))),
    )
    monkeypatch.setattr(
        server_module,
        "setup_otel_tracing_early",
        lambda **kwargs: events.append(("otel", kwargs["service_name"])),
    )
    monkeypatch.setattr(
        server_module, "build_service", lambda settings: events.append(("build",)) or object()
    )
    monkeypatch.setattr(server_module, "KnowledgeReadinessProbe", lambda settings: object())
    monkeypatch.setattr(
        server_module,
        "create_oracle_knowledge_server",
        lambda *args, **kwargs: events.append(("server",)) or FakeServer(),
    )
    server_module.main()

    assert [event[0] for event in events] == ["logging", "otel", "build", "server", "run"]
    assert events[0][1] == "oracle-knowledge-mcp"
    assert events[1][1] == "oracle-knowledge-mcp"

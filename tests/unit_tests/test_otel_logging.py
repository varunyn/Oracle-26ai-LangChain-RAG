import logging

import pytest

from src.rag_agent.utils import logging_config as lc


def reset_logging_state():
    """Reset module-level flag and detach handlers added by setup for isolated tests."""
    # Reset configured flag
    lc._configured = False  # type: ignore[attr-defined]

    # Remove our handlers/filters from root to avoid cross-test interference
    root = logging.getLogger()
    # Keep a copy to avoid modifying while iterating
    to_remove: list[logging.Handler] = list(root.handlers)
    for h in to_remove:
        root.removeHandler(h)
    for f in list(getattr(root, "filters", [])):
        try:
            root.removeFilter(f)
        except Exception:
            pass

    # Reset uvicorn logger handlers/propagation to defaults for test isolation
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


@pytest.fixture(autouse=True)
def _isolate_logging():
    reset_logging_state()
    yield
    reset_logging_state()


def test_otlp_handler_installed_and_idempotent():
    # First setup
    lc.setup_logging(console=False)
    root = logging.getLogger()

    # Exactly one LoggingHandler present
    from opentelemetry.sdk._logs import LoggingHandler

    handlers = [h for h in root.handlers if isinstance(h, LoggingHandler)]
    assert len(handlers) == 1

    # Second setup is a no-op
    lc.setup_logging(console=False)
    handlers2 = [h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler)]
    assert len(handlers2) == 1


def test_request_id_injection(caplog: pytest.LogCaptureFixture):
    lc.setup_logging(console=False)
    lc.set_request_id("req-abc123")

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("api.test")
    logger.info("hello world")

    # Ensure at least one captured record has injected request_id
    assert any(getattr(r, "request_id", None) == "req-abc123" for r in caplog.records)


def test_fail_open_when_exporter_errors(monkeypatch):
    # Force exporter to raise during export; ensure logging call doesn't raise
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    def boom(self, batch):  # type: ignore[no-untyped-def]
        raise RuntimeError("export failed")

    monkeypatch.setattr(OTLPLogExporter, "export", boom, raising=True)

    lc.setup_logging(console=True)

    logger = logging.getLogger("src.rag_agent.test")
    # Should not raise even if exporter fails; console handler still processes
    logger.info("this should not crash if collector is down")


@pytest.mark.parametrize("uv_name", ["uvicorn", "uvicorn.error", "uvicorn.access"])
def test_uvicorn_loggers_propagate_to_root(uv_name: str):
    lc.setup_logging(console=False)
    uv_logger = logging.getLogger(uv_name)
    assert uv_logger.propagate is True
    # We clear handlers so they bubble to root
    assert len(uv_logger.handlers) == 0

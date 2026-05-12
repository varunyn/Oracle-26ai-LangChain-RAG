import logging
from types import SimpleNamespace

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


def test_logging_analytics_query_event_filter_handles_otlp_log_data_wrappers():
    wrapped = SimpleNamespace(
        log_record=SimpleNamespace(
            body="chat_out answer_len=12 error=None",
            severity_number=9,
        )
    )

    assert lc._is_query_event_record(wrapped) is True  # type: ignore[arg-type]
    assert lc._severity_number_from_record(wrapped) == 9  # type: ignore[arg-type]


def test_request_id_filter_flattens_otel_attributes_for_logging_handler():
    record = logging.LogRecord(
        name="api.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="chat_out answer_len=12",
        args=(),
        exc_info=None,
    )
    record.otel_attributes = {  # type: ignore[attr-defined]
        "event_type": "chat_out",
        "answer_len": 12,
        "mcp_used": False,
        "ignored_complex": {"nested": True},
    }

    assert lc.RequestIdFilter().filter(record) is True
    assert record.event_type == "chat_out"  # type: ignore[attr-defined]
    assert record.answer_len == 12  # type: ignore[attr-defined]
    assert record.mcp_used is False  # type: ignore[attr-defined]
    assert not hasattr(record, "ignored_complex")
    assert not hasattr(record, "attributes")
    assert not hasattr(record, "otel_attributes")


def test_normalize_otlp_json_flattens_nested_attribute_lists_from_log_records():
    payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "severityNumber": "SEVERITY_NUMBER_INFO",
                                "attributes": [
                                    {
                                        "key": "attributes",
                                        "value": {
                                            "stringValue": (
                                                "[{key=event_type, value={stringValue=chat_out}}, "
                                                "{key=answer_len, value={intValue=285}}, "
                                                "{key=mcp_used, value={boolValue=false}}, "
                                                "{key=mcp_tool_names, value={stringValue=}}]"
                                            )
                                        },
                                    },
                                    {
                                        "key": "otel_attributes",
                                        "value": {
                                            "stringValue": (
                                                "[{key=event_type, value={stringValue=chat_out}}, "
                                                "{key=answer_len, value={intValue=285}}]"
                                            )
                                        },
                                    },
                                    {"key": "request_id", "value": {"stringValue": "req-1"}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    lc._normalize_otlp_json_for_oci(payload)

    attrs = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]
    by_key = {attr["key"]: attr["value"] for attr in attrs}
    assert "attributes" not in by_key
    assert "otel_attributes" not in by_key
    assert by_key["event_type"] == {"stringValue": "chat_out"}
    assert by_key["answer_len"] == {"intValue": "285"}
    assert by_key["mcp_used"] == {"boolValue": False}
    assert by_key["mcp_tool_names"] == {"stringValue": ""}
    assert by_key["request_id"] == {"stringValue": "req-1"}


def test_normalize_otlp_json_drops_attributes_without_values():
    payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "attributes": [
                                    {"key": "event_type", "value": {"stringValue": "chat_out"}},
                                    {"key": "error"},
                                    {"key": "code.function.name"},
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    lc._normalize_otlp_json_for_oci(payload)

    attrs = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["attributes"]
    by_key = {attr["key"]: attr["value"] for attr in attrs}
    assert by_key == {"event_type": {"stringValue": "chat_out"}}

"""
Centralized logging configuration for the RAG app (FastAPI + Python best practices).

OTLP-only logging (no local rotating files):
- Exports Python logs via OpenTelemetry Logs over OTLP HTTP to a local collector
- Optional console stream handler for local development visibility
- Request ID is stored in contextvars and injected into every log record for traceability
- Idempotent: repeated calls to setup_logging() do nothing
- Fail-open: if exporter/collector is unavailable, the app still starts and console logging works
- Uvicorn loggers propagate to root so our handlers/processors are applied

Env vars honored (defaults shown):
- OTEL_EXPORTER_OTLP_LOGS_ENDPOINT (default: http://localhost:4318/v1/logs)
- OTEL_EXPORTER_OTLP_ENDPOINT       (used as base when LOGS endpoint not set)
- OTEL_EXPORTER_OTLP_HEADERS / OTEL_EXPORTER_OTLP_LOGS_HEADERS (exporter will honor if provided)
- OTEL_SERVICE_NAME (default: rag-api)
- ENABLE_OCI_LOGGING_ANALYTICS / LOGGING_ANALYTICS_* (optional dual-ship to Oracle Logging Analytics)
- LOGGING_ANALYTICS_MIN_LEVEL — optional; by default OCI receives (1) user query events (flow_trace, chat_out) and (2) all WARNING+
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import threading
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from google.protobuf.json_format import MessageToJson  # type: ignore[import-untyped]
from oci import config as oci_config  # type: ignore[import-untyped]
from oci.log_analytics import LogAnalyticsClient  # type: ignore[import-untyped]
from oci.signer import Signer  # type: ignore[import-untyped]
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.common._internal._log_encoder import (
    encode_logs,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.proto.logs.v1.logs_pb2 import SeverityNumber
from opentelemetry.sdk._logs import (
    LoggerProvider as SDKLoggerProvider,
)
from opentelemetry.sdk._logs import (
    LoggingHandler,
    ReadableLogRecord,
)
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.resources import Resource

# Context variable for request-scoped request ID (set by middleware)
REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

logger = logging.getLogger(__name__)

_configured = False

_TRUTHY_STRINGS = {"1", "true", "yes", "on"}

LOGGING_ANALYTICS_MODE_AUTO = "auto"
LOGGING_ANALYTICS_MODE_ALL = "all"

_EXPORT_CONFIRMATION_PREFIX = "OCI Logging Analytics exported batch"

# Log message prefixes we treat as "user query" events (one per request: route, mode, success/error).
_QUERY_EVENT_PREFIXES = ("flow_trace ", "flow_trace\t", "chat_out ", "chat_out\t")


def _get_logging_analytics_mode() -> str:
    value = os.getenv("LOGGING_ANALYTICS_MODE")
    if value:
        value = value.strip().lower()
        if value in {LOGGING_ANALYTICS_MODE_AUTO, LOGGING_ANALYTICS_MODE_ALL}:
            return value
    try:
        from src.rag_agent.core import config as _cfg

        attr = getattr(_cfg, "LOGGING_ANALYTICS_MODE", LOGGING_ANALYTICS_MODE_AUTO)
    except Exception:  # noqa: BLE001
        attr = LOGGING_ANALYTICS_MODE_AUTO
    mode = str(attr).strip().lower()
    if mode not in {LOGGING_ANALYTICS_MODE_AUTO, LOGGING_ANALYTICS_MODE_ALL}:
        mode = LOGGING_ANALYTICS_MODE_AUTO
    return mode


_SEVERITY_WARN = SeverityNumber.SEVERITY_NUMBER_WARN  # 13


def _get_record_body_str(record: ReadableLogRecord) -> str:
    """Return the log record body as a string for pattern matching."""
    body = record.get("body") if isinstance(record, dict) else getattr(record, "body", None)
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, dict) and "stringValue" in body:
        return str(body["stringValue"])
    return str(body)


def _is_query_event_record(record: ReadableLogRecord) -> bool:
    """True if this log is a per-request summary (flow_trace or chat_out) to send to OCI."""
    s = _get_record_body_str(record)
    return any(s.startswith(p) for p in _QUERY_EVENT_PREFIXES)


def _severity_number_from_record(record: ReadableLogRecord) -> int:
    """Return OTel severity_number (int) for the record, or INFO (9) if unknown."""
    sn = (
        record.get("severity_number")
        if isinstance(record, dict)
        else getattr(record, "severity_number", None)
    )
    if sn is None:
        sn = (
            record.get("severityNumber")
            if isinstance(record, dict)
            else getattr(record, "severityNumber", None)
        )
    if isinstance(sn, int):
        return sn
    if isinstance(sn, str) and sn.startswith("SEVERITY_NUMBER_"):
        try:
            from typing import cast

            return cast(int, SeverityNumber.Value(sn))
        except (ValueError, TypeError):
            pass
    return SeverityNumber.SEVERITY_NUMBER_INFO  # default


def _is_export_confirmation_record(record: ReadableLogRecord) -> bool:
    """True if this log record is our own export confirmation (avoid sending it to OCI)."""
    return _EXPORT_CONFIRMATION_PREFIX in _get_record_body_str(record)


def _normalize_otlp_json_for_oci(obj: dict[str, object]) -> None:
    """Mutate OTLP JSON so OCI Logging Analytics accepts it (InvalidJsonFormat fix).

    OCI expects severityNumber as a number (e.g. 10); protobuf MessageToJson emits
    enum names (e.g. \"SEVERITY_NUMBER_INFO\"). Convert enum-name strings to int.
    See: https://docs.oracle.com/en-us/iaas/log-analytics/doc/upload-opentelemetry-logs.html
    """
    resource_logs = obj.get("resourceLogs")
    if not isinstance(resource_logs, list):
        return
    for resource_log in resource_logs:
        if not isinstance(resource_log, dict):
            continue
        scope_logs = resource_log.get("scopeLogs")
        if not isinstance(scope_logs, list):
            continue
        for scope_log in scope_logs:
            if not isinstance(scope_log, dict):
                continue
            log_records = scope_log.get("logRecords")
            if not isinstance(log_records, list):
                continue
            for record in log_records:
                if not isinstance(record, dict):
                    continue
                sn = record.get("severityNumber")
                if isinstance(sn, str) and sn.startswith("SEVERITY_NUMBER_"):
                    try:
                        record["severityNumber"] = SeverityNumber.Value(sn)
                    except (ValueError, TypeError):
                        pass
                # Ensure timestamps are strings (OCI/OTLP JSON spec)
                for key in ("timeUnixNano", "observedTimeUnixNano"):
                    if key in record and isinstance(record[key], (int, float)):
                        record[key] = str(int(record[key]))
                # Attribute intValue as string per OCI doc
                attributes = record.get("attributes")
                if not isinstance(attributes, list):
                    continue
                for attr in attributes:
                    if not isinstance(attr, dict):
                        continue
                    val = attr.get("value") or {}
                    if "intValue" in val and isinstance(val["intValue"], (int, float)):
                        val["intValue"] = str(int(val["intValue"]))


def _is_truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY_STRINGS


def _load_config_attr(name: str, default: str | None = None) -> str | None:
    try:
        from src.rag_agent.core import config as _cfg

        value = getattr(_cfg, name, None)
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip() or default
        return str(value)
    except Exception:  # noqa: BLE001
        return default


def _load_bool_config(name: str) -> bool:
    try:
        from src.rag_agent.core import config as _cfg

        value = getattr(_cfg, name, None)
        return _is_truthy(value)
    except Exception:  # noqa: BLE001
        return False


def _min_severity_from_level_name(name: str) -> int:
    """Map level name to OTel severity_number (DEBUG=5, INFO=9, WARNING=13, ERROR=17)."""
    level_map = {
        "DEBUG": SeverityNumber.SEVERITY_NUMBER_DEBUG,
        "INFO": SeverityNumber.SEVERITY_NUMBER_INFO,
        "WARNING": SeverityNumber.SEVERITY_NUMBER_WARN,
        "WARN": SeverityNumber.SEVERITY_NUMBER_WARN,
        "ERROR": SeverityNumber.SEVERITY_NUMBER_ERROR,
    }
    return level_map.get(name, SeverityNumber.SEVERITY_NUMBER_WARN)


@dataclass
class LoggingAnalyticsSettings:
    """Runtime settings required to upload OTLP logs to OCI Logging Analytics."""

    namespace: str
    log_group_id: str
    log_set: str | None
    meta_properties: str | None
    resource_category: str | None
    oci_profile: str
    oci_config_file: str
    min_severity_number: int = (
        SeverityNumber.SEVERITY_NUMBER_WARN
    )  # used if we add level-only mode; default sends query events + WARNING+


@lru_cache(maxsize=1)
def _get_logging_analytics_settings() -> LoggingAnalyticsSettings | None:
    if not _load_bool_config("ENABLE_OCI_LOGGING_ANALYTICS") and not _is_truthy(
        os.getenv("ENABLE_OCI_LOGGING_ANALYTICS")
    ):
        return None

    namespace = os.getenv("LOGGING_ANALYTICS_NAMESPACE") or _load_config_attr(
        "LOGGING_ANALYTICS_NAMESPACE"
    )
    log_group_id = os.getenv("LOGGING_ANALYTICS_LOG_GROUP_ID") or _load_config_attr(
        "LOGGING_ANALYTICS_LOG_GROUP_ID"
    )
    if not namespace or not log_group_id:
        logger.warning("ENABLE_OCI_LOGGING_ANALYTICS is True but namespace/log group missing")
        return None

    log_set = os.getenv("LOGGING_ANALYTICS_LOG_SET") or _load_config_attr(
        "LOGGING_ANALYTICS_LOG_SET"
    )
    meta_properties = os.getenv("LOGGING_ANALYTICS_META_PROPERTIES") or _load_config_attr(
        "LOGGING_ANALYTICS_META_PROPERTIES"
    )
    resource_category = os.getenv("LOGGING_ANALYTICS_RESOURCE_CATEGORY") or _load_config_attr(
        "LOGGING_ANALYTICS_RESOURCE_CATEGORY", "rag-api"
    )

    profile = os.getenv("OCI_PROFILE") or _load_config_attr("OCI_PROFILE", "DEFAULT")
    config_file = os.getenv("OCI_CONFIG_FILE") or _load_config_attr(
        "OCI_CONFIG_FILE", "~/.oci/config"
    )
    min_level = (
        os.getenv("LOGGING_ANALYTICS_MIN_LEVEL")
        or _load_config_attr("LOGGING_ANALYTICS_MIN_LEVEL")
        or "WARNING"
    )
    min_severity = _min_severity_from_level_name(str(min_level).strip().upper())
    return LoggingAnalyticsSettings(
        namespace=namespace,
        log_group_id=log_group_id,
        log_set=log_set,
        meta_properties=meta_properties,
        resource_category=resource_category,
        oci_profile=profile or "DEFAULT",
        oci_config_file=config_file or "~/.oci/config",
        min_severity_number=min_severity,
    )


class LoggingAnalyticsExporter(LogRecordExporter):
    """Custom exporter that uploads OTLP JSON payloads to OCI Logging Analytics."""

    def __init__(self, settings: LoggingAnalyticsSettings, service_name: str) -> None:
        self._settings = settings
        self._service_name = service_name or "rag-api"
        self._client = self._build_client(settings)
        self._lock = threading.Lock()

    def _build_client(self, settings: LoggingAnalyticsSettings) -> LogAnalyticsClient:
        config_path = Path(settings.oci_config_file).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"OCI config file not found: {config_path}")
        # Load config ourselves so we can resolve key_file relative to config file dir (SDK validates path in from_file before we can fix it).
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(config_path)
        if settings.oci_profile not in parser:
            raise ValueError(f"Profile '{settings.oci_profile}' not found in {config_path}")
        cfg = dict(oci_config.DEFAULT_CONFIG)
        cfg.update(dict(parser[settings.oci_profile]))
        log_requests_val = cfg.get("log_requests", "false")
        cfg["log_requests"] = str(log_requests_val).lower() in (
            "1",
            "yes",
            "true",
            "on",
        )
        key_file = cfg.get("key_file")
        if isinstance(key_file, str) and key_file:
            key_path = Path(key_file).expanduser()
            if not key_path.is_absolute():
                key_path = (config_path.parent / key_path).resolve()
            cfg["key_file"] = str(key_path)
        signer = Signer.from_config(cfg)
        client = LogAnalyticsClient(cfg, signer=signer)
        return client

    def _meta_properties(self) -> str | None:
        if self._settings.meta_properties:
            return self._settings.meta_properties
        pieces: list[str] = []
        if self._service_name:
            pieces.append(f"sourceName:{self._service_name}")
        if self._settings.resource_category:
            pieces.append(f"resourceCategory:{self._settings.resource_category}")
        return ";".join(pieces) if pieces else None

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        mode = _get_logging_analytics_mode()
        batch = [r for r in batch if not _is_export_confirmation_record(r)]
        if mode == "auto":
            batch = [
                r
                for r in batch
                if _is_query_event_record(r) or _severity_number_from_record(r) >= _SEVERITY_WARN
            ]
        if not batch:
            return LogRecordExportResult.SUCCESS
        try:
            # OTLP JSON format per Oracle Log Analytics and OTel spec (resourceLogs / scopeLogs / logRecords).
            payload = encode_logs(batch)
            json_str = MessageToJson(payload)
            # OCI expects severityNumber as number and specific formats; normalize to avoid InvalidJsonFormat.
            payload_dict = json.loads(json_str)
            _normalize_otlp_json_for_oci(payload_dict)
            json_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
            # Use "application/json; charset=utf-8" so OCI base_client does not run
            # sanitize_for_serialization on the body (it only skips for content_type != "application/json").
            kwargs: dict[str, str] = {
                "content_type": "application/json; charset=utf-8",
                "opc_request_id": str(uuid.uuid4()),
            }
            meta = self._meta_properties()
            if meta:
                kwargs["opc_meta_properties"] = meta
            if self._settings.log_set:
                kwargs["log_set"] = self._settings.log_set
            with self._lock:
                self._client.upload_otlp_logs(  # type: ignore[call-arg]
                    namespace_name=self._settings.namespace,
                    opc_meta_loggrpid=self._settings.log_group_id,
                    upload_otlp_logs_details=json_bytes,
                    **kwargs,
                )
            return LogRecordExportResult.SUCCESS
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OCI Logging Analytics export failed (check region/namespace/log group OCID and IAM): %s",
                exc,
                exc_info=True,
            )
            return LogRecordExportResult.FAILURE

    def shutdown(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


class RequestIdFilter(logging.Filter):
    """Inject request_id and optional OpenTelemetry attributes into records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.request_id = REQUEST_ID_CTX.get()  # type: ignore[attr-defined]
        otel_attrs = getattr(record, "otel_attributes", None)
        if isinstance(otel_attrs, dict):
            record.attributes = otel_attrs  # type: ignore[attr-defined]
        return True


def _compute_logs_endpoint() -> str:
    # Prefer config.OTEL_LOGS_ENDPOINT, then env OTEL_EXPORTER_OTLP_LOGS_ENDPOINT, then base + /v1/logs
    try:
        from src.rag_agent.core import config as _cfg

        ep = getattr(_cfg, "OTEL_LOGS_ENDPOINT", None)
        if isinstance(ep, str) and ep.strip():
            return ep.strip()
    except Exception:  # noqa: BLE001
        pass
    logs_ep = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if logs_ep:
        return logs_ep
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base:
        base = base.rstrip("/")
        return f"{base}/v1/logs"
    return "http://localhost:4318/v1/logs"


def setup_logging(console: bool = True) -> None:
    """
    Configure application logging to export via OTLP HTTP Logs and optional console stream.

    - Adds a single OpenTelemetry LoggingHandler bound to a LoggerProvider with a
      BatchLogRecordProcessor + OTLP HTTP exporter
    - Adds RequestIdFilter at root-logger level so every record carries request_id
    - Ensures uvicorn loggers propagate to root and don't attach their own handlers
    - Safe to call multiple times; only first call applies
    """
    global _configured
    if _configured:
        return

    # Suppress OCI SDK and urllib3 so their HTTP/DEBUG logs are not captured and re-exported to OCI.
    for _logger_name in (
        "oci",
        "urllib3",
        "oci._vendor.urllib3",
        "langsmith",
        "langsmith._internal",
        "langsmith._internal.otel",
        "langsmith._internal.otel._otel_exporter",
    ):
        logging.getLogger(_logger_name).setLevel(logging.WARNING)

    # Build resource (service.name can be overridden via env)
    service_name = os.getenv("OTEL_SERVICE_NAME", "rag-api")
    attribute_mapping = [
        {"attributeName": "event_type", "laFieldName": "Event Type"},
        {"attributeName": "route", "laFieldName": "Route"},
        {"attributeName": "answer_source", "laFieldName": "Answer Source"},
        {"attributeName": "answer_len", "laFieldName": "Answer Length"},
        {"attributeName": "node_name", "laFieldName": "Node Name"},
        {"attributeName": "duration_ms", "laFieldName": "Duration MS"},
        {"attributeName": "error", "laFieldName": "Error"},
    ]
    resource = Resource.create(
        {
            "service.name": service_name,
            "oci_la_attribute_mapping": json.dumps(attribute_mapping),
        }
    )

    # Create logger provider and exporter (fail-open if exporter init fails)
    provider = SDKLoggerProvider(resource=resource)

    exporters: list[LogRecordExporter] = []
    exporter_descriptions: list[str] = []
    logs_endpoint = _compute_logs_endpoint()
    # Default endpoint: use 30s timeout so collector→Loki pipeline can complete (Loki can respond in ~15s); when no collector runs, fail after 30s.
    default_logs = "http://localhost:4318/v1/logs"
    otlp_timeout = 30.0 if logs_endpoint == default_logs else None
    try:
        exporters.append(OTLPLogExporter(endpoint=logs_endpoint, timeout=otlp_timeout))
        desc = f"OTLP ({logs_endpoint}"
        if otlp_timeout is not None:
            desc += f", timeout={otlp_timeout}s"
        desc += ")"
        exporter_descriptions.append(desc)
    except Exception as exc:  # Fail-open: continue without OTLP exporter
        logger.warning("OTLP log exporter init failed: %s", exc)

    la_settings = _get_logging_analytics_settings()
    if la_settings:
        try:
            exporters.append(LoggingAnalyticsExporter(la_settings, service_name))
            exporter_descriptions.append(
                f"OCI Logging Analytics (namespace={la_settings.namespace})"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enable Logging Analytics exporter: %s", exc)

    if not exporters:
        logger.warning("No log exporters configured; console-only logging active")

    for exp in exporters:
        provider.add_log_record_processor(BatchLogRecordProcessor(exp))

    # Register provider globally so OpenTelemetry-aware libs can find it
    set_logger_provider(provider)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Ensure request_id is injected for ALL records (works with caplog and any handler)
    # Avoid adding duplicate filters if called from test harness without process restart
    if not any(isinstance(f, RequestIdFilter) for f in getattr(root_logger, "filters", [])):
        root_logger.addFilter(RequestIdFilter())

    # Add a single LoggingHandler (bridges stdlib logging -> OTel Logs)
    if not any(isinstance(h, LoggingHandler) for h in root_logger.handlers):
        otel_handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
        # The filter ensures record has request_id attribute before export
        otel_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(otel_handler)

    # Optional console stream for local visibility
    if console and not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(RequestIdFilter())
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Log exporter summary so it appears on console (handler was just added above)
    if exporters:
        logger.info("Log exporters: %s", "; ".join(exporter_descriptions))

    # Ensure uvicorn and uvicorn.access loggers propagate to root so our format/handlers are used
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # Avoid flooding OTLP/Loki with HTTP client logs (every POST to /v1/logs and /v1/traces)
    for name in (
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.connection",
        "requests",
        "opentelemetry.exporter",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    # aiosqlite can log every DB op at DEBUG including full
    # checkpoint payloads; silence to avoid huge log lines.
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    # Downgrade SDK export-failure tracebacks (e.g. ReadTimeout when collector is down) to avoid console flood
    logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.WARNING)
    for name in (
        "mcp_use",
        "mcp_use.telemetry",
        "mcp_use.telemetry.telemetry",
        "mcp_use.client",
        "mcp_use.client.connectors",
        "mcp_use.client.task_managers",
        "mcp_use.client.task_managers.streamable_http",
        "mcp_use.client.middleware",
        "mcp",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def set_request_id(value: str) -> None:
    """Set the current request ID (e.g. from middleware)."""
    REQUEST_ID_CTX.set(value)


def get_request_id() -> str:
    """Return the current request ID or '-' if not set."""
    return REQUEST_ID_CTX.get()

"""
OpenTelemetry tracing bootstrap for FastAPI and LangChain/LangGraph.

Idempotent; exports via OTLP HTTP (default http://localhost:4318/v1/traces);
sets Resource service.name=rag-api; ENABLE_OTEL_TRACING gates; fail-open.

When ENABLE_OTEL_TRACING is on, call setup_otel_tracing_early() before importing
any LangChain/LangGraph code so graph invokes are traced; then call
setup_otel_tracing(app) in lifespan to add FastAPI/Requests instrumentation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

# OTLP HTTP exporter (installed via opentelemetry-exporter-otlp-proto-http)
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,  # type: ignore
    )
except Exception:  # pragma: no cover - ImportError path covered in tests indirectly
    OTLPSpanExporter = None  # type: ignore

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = logging.getLogger(__name__)

_INITIALIZED_LOCK = threading.Lock()
_INITIALIZED = False
# Set by setup_otel_tracing_early() so setup_otel_tracing(app) reuses the same provider
_EARLY_PROVIDER: TracerProvider | None = None

_DEFAULT_TRACES_ENDPOINT = "http://localhost:4318/v1/traces"


def _otel_safe_attribute_value(value: Any) -> Any:
    """Convert a value to an OTEL-allowed type (bool, str, int, float, or sequence of same)."""
    if value is None:
        return None
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(type(x) in (bool, str, int, float) for x in value):
            return list(value)
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _patch_langsmith_otel_metadata() -> None:
    """Patch LangSmith OTEL exporter so metadata dict values are serialized; avoids 'Invalid type dict' warnings."""
    try:
        from langsmith._internal.otel import _otel_exporter as _otel_mod
    except ImportError:
        return
    orig = _otel_mod.OTELExporter._set_span_attributes

    def _set_span_attributes_patched(
        self: Any, span: Any, run_info: dict[str, Any], op: Any
    ) -> None:
        extra = run_info.get("extra") or {}
        metadata = extra.get("metadata") or {}
        if metadata:
            new_meta: dict[str, Any] = {}
            for k, v in metadata.items():
                if v is None:
                    continue
                safe = _otel_safe_attribute_value(v)
                if safe is not None:
                    new_meta[k] = safe
            if new_meta:
                run_info = dict(run_info)
                run_info["extra"] = dict(extra)
                run_info["extra"]["metadata"] = new_meta
        orig(self, span, run_info, op)

    setattr(_otel_mod.OTELExporter, "_set_span_attributes", _set_span_attributes_patched)
    _logger.debug("LangSmith OTEL exporter patched to sanitize metadata attributes")


def _env_enabled(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is not None:
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    try:
        from api.settings import get_settings

        return bool(get_settings().ENABLE_OTEL_TRACING)
    except Exception:  # noqa: BLE001
        pass
    return default


def _get_traces_endpoint() -> str:
    """Return OTLP traces endpoint: config.OTEL_TRACES_ENDPOINT, then env, then default."""
    try:
        import config as _cfg  # type: ignore

        ep = getattr(_cfg, "OTEL_TRACES_ENDPOINT", None)
        if isinstance(ep, str) and ep.strip():
            return ep.strip()
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", _DEFAULT_TRACES_ENDPOINT)


def _get_traces_headers() -> dict[str, str] | None:
    """Return optional HTTP headers for trace exporter (e.g. Oracle APM dataKey auth)."""
    try:
        import config as _cfg  # type: ignore

        h = getattr(_cfg, "OTEL_TRACES_HEADERS", None)
        if isinstance(h, dict) and h:
            return {str(k): str(v) for k, v in h.items()}
    except Exception:  # noqa: BLE001
        pass
    return None


def setup_otel_tracing_early() -> bool:
    """Create TracerProvider and set it globally before any LangChain/LangGraph import.

    Call this in api/main.py before importing routers so agent_graph.invoke() and
    all LangChain runnables emit spans to our OTLP collector. Also sets
    LANGSMITH_OTEL_ENABLED / LANGSMITH_TRACING / LANGSMITH_OTEL_ONLY so LangSmith
    uses this provider and sends only to our endpoint (no LangSmith API key needed).

    Returns True if the provider was set in this call, False if disabled or already set.
    """
    global _EARLY_PROVIDER

    if not _env_enabled("ENABLE_OTEL_TRACING", default=False):
        _logger.debug("OTel tracing disabled; skipping early setup")
        return False

    with _INITIALIZED_LOCK:
        if _EARLY_PROVIDER is not None:
            _logger.debug("OTel early provider already set; skipping")
            return False
        try:
            # LangSmith: use our TracerProvider and send only to OTLP (no LangSmith API)
            os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_OTEL_ONLY"] = "true"

            resource = Resource.create({SERVICE_NAME: "rag-api"})
            # shutdown_on_exit=False avoids blocking process exit when OTLP endpoint is slow/unreachable
            provider = TracerProvider(resource=resource, shutdown_on_exit=False)
            if OTLPSpanExporter is None:
                raise RuntimeError(
                    "opentelemetry-exporter-otlp-proto-http is not installed; cannot create OTLP exporter"
                )
            endpoint = _get_traces_endpoint()
            # Skip export when endpoint is default localhost:4318 to avoid ReadTimeout when no collector runs.
            if endpoint.strip().rstrip("/") == _DEFAULT_TRACES_ENDPOINT.strip().rstrip("/"):
                _logger.info(
                    "OTel tracing enabled but endpoint is default localhost:4318; skipping trace exporter (start observability stack or set OTEL_TRACES_ENDPOINT for OCI APM)"
                )
            else:
                headers = _get_traces_headers()
                exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers if headers else None)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _EARLY_PROVIDER = provider
            _patch_langsmith_otel_metadata()
            _logger.info("OpenTelemetry early provider set (LangChain/LangGraph will be traced)")
            return True
        except Exception as e:  # noqa: BLE001
            _logger.warning("Failed to set OTel early provider (non-fatal): %s", e)
            return False


def setup_otel_tracing(app: FastAPI | None = None, exporter: SpanExporter | None = None) -> bool:
    """Initialize or complete OpenTelemetry tracing (FastAPI + Requests instrumentation).

    If setup_otel_tracing_early() was called, reuses that provider and only adds
    FastAPI/Requests instrumentation. Otherwise creates provider and instrumentation.

    Returns True if initialization occurred in this call, False if skipped.
    """
    global _INITIALIZED

    if not _env_enabled("ENABLE_OTEL_TRACING", default=False) and exporter is None:
        _logger.debug("OTel tracing disabled by env (ENABLE_OTEL_TRACING not truthy)")
        return False

    with _INITIALIZED_LOCK:
        if _INITIALIZED:
            _logger.debug("OTel tracing already initialized; skipping")
            return False
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.instrumentation.requests import RequestsInstrumentor

            provider = _EARLY_PROVIDER
            if provider is None:
                # No early setup: create provider and exporter
                resource = Resource.create({SERVICE_NAME: "rag-api"})
                provider = TracerProvider(resource=resource, shutdown_on_exit=False)
                trace.set_tracer_provider(provider)
                if exporter is None:
                    endpoint = _get_traces_endpoint()
                    if endpoint.strip().rstrip("/") != _DEFAULT_TRACES_ENDPOINT.strip().rstrip("/"):
                        if OTLPSpanExporter is None:
                            raise RuntimeError(
                                "opentelemetry-exporter-otlp-proto-http is not installed"
                            )
                        headers = _get_traces_headers()
                        exporter = OTLPSpanExporter(
                            endpoint=endpoint, headers=headers if headers else None
                        )
                        provider.add_span_processor(BatchSpanProcessor(exporter))
                else:
                    provider.add_span_processor(BatchSpanProcessor(exporter))

            # If using early provider and exporter provided, add it
            if provider is _EARLY_PROVIDER and exporter is not None:
                provider.add_span_processor(BatchSpanProcessor(exporter))

            if app is not None:
                try:
                    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
                except Exception as e:  # noqa: BLE001
                    _logger.warning("FastAPI instrumentation failed (non-fatal): %s", e)
            else:
                try:
                    FastAPIInstrumentor().instrument()
                except Exception as e:  # noqa: BLE001
                    _logger.debug("Global FastAPI instrumentation failed (non-fatal): %s", e)

            try:
                RequestsInstrumentor().instrument()
            except Exception as e:  # noqa: BLE001
                _logger.debug("Requests instrumentation failed (non-fatal): %s", e)

            _patch_langsmith_otel_metadata()
            _INITIALIZED = True
            _logger.info("OpenTelemetry tracing initialized (service.name=rag-api)")
            return True
        except Exception as e:  # noqa: BLE001
            _logger.warning("Failed to initialize OpenTelemetry tracing (non-fatal): %s", e)
            return False

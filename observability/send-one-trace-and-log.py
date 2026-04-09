#!/usr/bin/env python3
"""
Send one trace (span) and one log to the OTLP collector for E2E verification.

Usage (from project root, with observability stack up):
  uv run python observability/send-one-trace-and-log.py

Then run observability/check-data.sh or query Grafana (Loki: {}, Tempo: service.name=rag-api).
"""

from __future__ import annotations

import logging
import os
import sys

import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_TRACES_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"
)
_LOGS_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://localhost:4318/v1/logs")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-api")


def main() -> int:
    resource = Resource.create({"service.name": _SERVICE_NAME})

    # Suppress exporter tracebacks; we'll exit with a clear message on connection errors
    logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)

    try:
        # One span (SimpleSpanProcessor so it exports immediately)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=_TRACES_ENDPOINT))
        )
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__, "1.0.0")
        with tracer.start_as_current_span("e2e-test-span") as span:
            span.set_attribute("e2e", "true")
        tracer_provider.shutdown()

        # One log (batch then flush via shutdown)
        logger_provider = SDKLoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=_LOGS_ENDPOINT))
        )
        handler = LoggingHandler(logger_provider=logger_provider)
        log = logging.getLogger("e2e-test")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        log.addHandler(handler)
        log.info("E2E test log message from send-one-trace-and-log.py")
        logger_provider.shutdown()
    except requests.exceptions.ConnectionError:
        print("Collector unreachable (connection refused). Start stack:", file=sys.stderr)
        print(
            "  docker compose --profile observability up -d loki tempo otel-collector grafana",
            file=sys.stderr,
        )
        print("Endpoints:", _TRACES_ENDPOINT, _LOGS_ENDPOINT, file=sys.stderr)
        return 1

    print(
        "Sent 1 span and 1 log to collector. Wait a few seconds then run: bash observability/check-data.sh"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Core logging shims for rag_agent.

This module re-exports stable logging symbols from rag_agent.utils.logging_config
so internal callers can depend on a core path that remains stable while the
underlying implementation lives under utils/.

Behavioral contract:
- Request ID correlation (REQUEST_ID_CTX, set_request_id, get_request_id)
- OTLP logs exporter configuration (fail-open semantics)
- No FastAPI or API runtime dependency
"""

from __future__ import annotations

from ..utils.logging_config import (
    REQUEST_ID_CTX,
    LoggingAnalyticsExporter,
    LoggingAnalyticsSettings,
    RequestIdFilter,
    get_request_id,
    set_request_id,
    setup_logging,
)

__all__ = [
    "REQUEST_ID_CTX",
    "setup_logging",
    "set_request_id",
    "get_request_id",
    "RequestIdFilter",
    "LoggingAnalyticsExporter",
    "LoggingAnalyticsSettings",
]

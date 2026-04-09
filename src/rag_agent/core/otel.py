"""
Core OTEL wrapper for rag_agent.

Re-exports stable API from utils.otel_tracing for backwards compatibility.
"""

from __future__ import annotations

from src.rag_agent.utils.otel_tracing import setup_otel_tracing, setup_otel_tracing_early

__all__ = ["setup_otel_tracing_early", "setup_otel_tracing"]

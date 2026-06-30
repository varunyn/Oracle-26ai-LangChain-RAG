"""
App-level resources created at startup and cleaned up at shutdown.

- Centralizes construction of long-lived FastAPI product-API resources
- Keeps request-time dependencies (FastAPI Depends) separate from resource wiring

This module has no import-time side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from api.settings import Settings, get_settings
from src.rag_agent.infrastructure.mcp_adapter_runtime import clear_adapter_runtime_cache
from src.rag_agent.utils.langfuse_tracing import safe_shutdown as langfuse_safe_shutdown


@dataclass
class AppResources:
    """Container for application-scoped resources."""

    settings: Settings


async def create_app_resources() -> AppResources:
    """Build and return application-scoped resources.

    Called once in FastAPI lifespan startup.
    """
    settings = get_settings()
    return AppResources(settings=settings)


async def shutdown_app_resources(resources: AppResources | None) -> None:
    """Tear down resources on application shutdown."""
    if not resources:
        return
    try:
        langfuse_safe_shutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        await clear_adapter_runtime_cache()
    except Exception:  # noqa: BLE001
        pass

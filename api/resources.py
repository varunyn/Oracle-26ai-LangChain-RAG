"""
App-level resources created at startup and cleaned up at shutdown.

- Centralizes construction of long-lived services (e.g., ChatRuntimeService)
- Keeps request-time dependencies (FastAPI Depends) separate from resource wiring

This module has no import-time side effects.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast

from api.services.graph_service import ChatRuntimeService
from api.settings import Settings, get_settings
from src.rag_agent.infrastructure.mcp_adapter_runtime import clear_adapter_runtime_cache
from src.rag_agent.utils.langfuse_tracing import safe_shutdown as langfuse_safe_shutdown


@dataclass
class AppResources:
    """Container for application-scoped resources."""

    settings: Settings
    chat_runtime_service: ChatRuntimeService
    _state_conn: object | None = None  # Reserved for optional durable-state backends.

    def get_state_conn(self) -> object | None:
        return self._state_conn


async def create_app_resources() -> AppResources:
    """Build and return application-scoped resources.

    Called once in FastAPI lifespan startup.
    """
    settings = get_settings()
    chat_runtime_service = ChatRuntimeService()
    return AppResources(
        settings=settings,
        chat_runtime_service=chat_runtime_service,
        _state_conn=None,
    )


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
    get_state_conn = getattr(resources, "get_state_conn", None)
    conn = get_state_conn() if callable(get_state_conn) else None
    if conn is not None:
        close_method = getattr(conn, "close", None)
        if callable(close_method):
            try:
                result_obj: object = close_method()
                if inspect.isawaitable(result_obj):
                    _ = await cast(Awaitable[object], result_obj)
            except Exception:  # noqa: BLE001
                pass
        object.__setattr__(resources, "_state_conn", None)

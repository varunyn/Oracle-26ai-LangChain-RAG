"""
App-level resources created at startup and cleaned up at shutdown.

- Centralizes construction of long-lived services (e.g., GraphService)
- Keeps request-time dependencies (FastAPI Depends) separate from resource wiring

This module has no import-time side effects.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from api.services.graph_service import GraphService
from api.settings import Settings, get_settings
from src.rag_agent.infrastructure.code_mode_client import (
    aclose_code_mode_client,
    init_code_mode_client,
)
from src.rag_agent.langgraph import graph as rag_graph


@dataclass
class AppResources:
    """Container for application-scoped resources."""

    settings: Settings
    graph_service: GraphService
    _checkpointer_conn: object | None = None  # aiosqlite.Connection; closed on shutdown

    def get_checkpointer_conn(self) -> object | None:
        return self._checkpointer_conn


async def create_app_resources() -> AppResources:
    """Build and return application-scoped resources.

    Called once in FastAPI lifespan startup. Uses AsyncSqliteSaver so that
    graph.astream() and checkpoint access work natively in async context.
    """
    settings = get_settings()
    saver, conn_obj = cast(tuple[object, object], await rag_graph.create_async_checkpointer())
    create_workflow_func = cast(Callable[..., object], rag_graph.create_workflow)
    graph = create_workflow_func(checkpointer=saver)
    graph_service = GraphService(graph=graph)
    if settings.CODE_MODE_ENABLED:
        await init_code_mode_client()
    return AppResources(
        settings=settings,
        graph_service=graph_service,
        _checkpointer_conn=conn_obj,
    )


async def shutdown_app_resources(resources: AppResources | None) -> None:
    """Tear down resources on application shutdown."""
    if not resources:
        return
    await aclose_code_mode_client()
    conn = resources.get_checkpointer_conn()
    if conn is not None:
        close_method = getattr(conn, "close", None)
        if callable(close_method):
            try:
                result_obj: object = close_method()
                if inspect.isawaitable(result_obj):
                    _ = await cast(Awaitable[object], result_obj)
            except Exception:  # noqa: BLE001
                pass
        object.__setattr__(resources, "_checkpointer_conn", None)

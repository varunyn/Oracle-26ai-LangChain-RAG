"""Request-scoped dependency providers.

Keep all FastAPI Depends providers that read from app.state here.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from api.resources import AppResources
from api.services.graph_service import GraphService
from api.settings import Settings
from api.settings import get_settings as get_settings_global

_fallback_graph_service: GraphService | None = None


def get_graph_service(request: Request) -> GraphService:
    """Provide GraphService from app.state.resources; fallback to singleton for non-ASGI usage."""
    resources = cast(
        AppResources | None, getattr(request.app.state, "resources", None)
    )  # pyright: ignore[reportAny]
    svc = resources.graph_service if resources else None
    if svc is not None:
        return svc
    global _fallback_graph_service
    if _fallback_graph_service is None:
        _fallback_graph_service = GraphService()
    return _fallback_graph_service


def get_settings(request: Request) -> Settings:
    """Provide Settings from app.state.resources; fallback to cached global Settings.

    Prefer app-scoped Settings created in lifespan to avoid duplicate instantiation.
    """
    resources = cast(
        AppResources | None, getattr(request.app.state, "resources", None)
    )  # pyright: ignore[reportAny]
    if resources and getattr(resources, "settings", None) is not None:
        return resources.settings
    return get_settings_global()

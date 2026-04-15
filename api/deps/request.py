"""Request-scoped dependency providers.

Keep all FastAPI Depends providers that read from app.state here.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from api.resources import AppResources
from api.services.graph_service import ChatRuntimeService
from api.settings import Settings
from api.settings import get_settings as get_settings_global


def _ensure_app_resources(request: Request) -> AppResources:
    resources = cast(
        AppResources | None, getattr(request.app.state, "resources", None)
    )  # pyright: ignore[reportAny]
    if resources is not None:
        return resources

    # Test/non-lifespan fallback: build minimal resources once and cache on app.state.
    resources = AppResources(
        settings=get_settings_global(),
        chat_runtime_service=ChatRuntimeService(),
        _state_conn=None,
    )
    request.app.state.resources = resources
    return resources


def get_graph_service(request: Request) -> ChatRuntimeService:
    """Provide ChatRuntimeService from app.state.resources."""
    resources = _ensure_app_resources(request)
    return resources.chat_runtime_service


def get_settings(request: Request) -> Settings:
    """Provide Settings from app.state.resources; fallback to cached global Settings.

    Prefer app-scoped Settings created in lifespan to avoid duplicate instantiation.
    """
    resources = _ensure_app_resources(request)
    if getattr(resources, "settings", None) is not None:
        return resources.settings
    return get_settings_global()

"""Request-scoped dependency providers for FastAPI product APIs."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from api.resources import AppResources
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
        _state_conn=None,
    )
    request.app.state.resources = resources
    return resources


def get_settings(request: Request) -> Settings:
    """Provide Settings from app.state.resources."""
    resources = _ensure_app_resources(request)
    return resources.settings

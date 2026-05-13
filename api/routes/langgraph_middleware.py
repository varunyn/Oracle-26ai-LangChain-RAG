"""Runtime middleware helpers shared by runtime route handlers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any


def resolve_request_id(headers: Mapping[str, str] | None) -> str:
    """Return incoming request id or a generated id."""
    if headers is None:
        return str(uuid.uuid4())
    request_id = headers.get("x-request-id") or headers.get("X-Request-ID")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    return str(uuid.uuid4())


def merge_runtime_context(
    *,
    top_level: dict[str, Any],
    context: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    configurable: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge runtime context maps while preserving explicit top-level values."""
    merged = dict(top_level)
    for container in (context, metadata, configurable):
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
    return merged

"""JSON serialization helpers for API responses and stream metadata."""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

try:
    import orjson as _orjson

    orjson: Any = _orjson
    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None  # type: ignore[assignment]
    ORJSON_AVAILABLE = False


def _json_default(obj: Any) -> Any:
    """Default serializer for Decimals and datetimes."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def safe_json(data: Any) -> str:
    """Serialize arbitrary payloads (used for SSE chunks)."""
    if ORJSON_AVAILABLE:
        return cast(
            str,
            orjson.dumps(
                data, default=_json_default, option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z
            ).decode("utf-8"),
        )
    return json.dumps(data, default=_json_default)


def make_metadata_safe(value: Any) -> Any:
    """Convert nested structures to JSON-safe types without re-encoding to str."""
    if isinstance(value, dict):
        return {k: make_metadata_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_metadata_safe(v) for v in value]
    if isinstance(value, (Decimal, datetime, date)):
        return _json_default(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

"""Helpers for LangChain/LangGraph v3 runtime event streams."""

from __future__ import annotations

import time


def v3_raw_event(*, method: str, data: object) -> dict[str, object]:
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": [],
            "timestamp": int(time.time() * 1000),
            "data": data,
        },
    }

from __future__ import annotations

from typing import cast


def to_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return max(int(float(stripped)), 0)
        except ValueError:
            return 0
    return 0


def normalize_usage(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage = cast(dict[str, object], raw)
    input_tokens = to_int(
        usage.get("input") or usage.get("prompt_tokens") or usage.get("input_tokens")
    )
    output_tokens = to_int(
        usage.get("output") or usage.get("completion_tokens") or usage.get("output_tokens")
    )
    total_tokens = to_int(usage.get("total") or usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


__all__ = [
    "to_int",
    "normalize_usage",
]

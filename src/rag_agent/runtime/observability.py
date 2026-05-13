from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float


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


def extract_usage(message: object) -> dict[str, int] | None:
    candidates: list[object] = []
    for attr in ("usage_metadata", "response_metadata", "additional_kwargs", "llm_output"):
        candidates.append(getattr(message, attr, None))
    if isinstance(message, dict):
        candidates.append(message)

    for candidate in candidates:
        parsed = normalize_usage(candidate)
        if parsed:
            return parsed
        if isinstance(candidate, dict):
            mapping = cast(dict[str, object], candidate)
            for key in ("usage", "token_usage", "usage_metadata"):
                parsed = normalize_usage(mapping.get(key))
                if parsed:
                    return parsed
    return None


def pricing_for_model(model_id: str | None, input_tokens: int) -> TokenPricing | None:
    model = (model_id or "").strip().lower()
    if not model:
        return None

    if "grok-code-fast-1" in model:
        return TokenPricing(input_per_million=0.20, output_per_million=1.50)
    if "grok-4.2" in model:
        if input_tokens > 200_000:
            return TokenPricing(input_per_million=4.00, output_per_million=12.00)
        return TokenPricing(input_per_million=2.00, output_per_million=6.00)
    if "grok-4-fast" in model:
        if input_tokens > 128_000:
            return TokenPricing(input_per_million=0.40, output_per_million=1.00)
        return TokenPricing(input_per_million=0.20, output_per_million=0.50)
    if "grok-3-mini-fast" in model:
        return TokenPricing(input_per_million=0.60, output_per_million=4.00)
    if "grok-3-fast" in model:
        return TokenPricing(input_per_million=5.00, output_per_million=25.00)
    if "grok-3-mini" in model:
        return TokenPricing(input_per_million=0.30, output_per_million=0.50)
    if "grok-3" in model or "grok-4" in model:
        return TokenPricing(input_per_million=3.00, output_per_million=15.00)

    if "gemini-2.5-pro" in model:
        if input_tokens > 200_000:
            return TokenPricing(input_per_million=2.50, output_per_million=15.00)
        return TokenPricing(input_per_million=1.25, output_per_million=10.00)
    if "gemini-2.5-flash-lite" in model:
        return TokenPricing(input_per_million=0.10, output_per_million=0.40)
    if "gemini-2.5-flash" in model:
        return TokenPricing(input_per_million=0.30, output_per_million=2.50)

    if "gpt-oss-120b" in model:
        return TokenPricing(input_per_million=0.15, output_per_million=0.60)
    if "gpt-oss-20b" in model:
        return TokenPricing(input_per_million=0.07, output_per_million=0.30)
    return None


def estimate_cost_usd(model_id: str | None, usage: dict[str, int]) -> tuple[float | None, str]:
    model = (model_id or "").strip().lower()
    if any(key in model for key in ("llama-4-scout", "llama-4-maverick", "large-meta")):
        return 0.0018 / 10_000.0, "transaction"
    if "llama-3.1-405b" in model:
        return 0.0267 / 10_000.0, "transaction"
    if any(key in model for key in ("llama-3.2-90b", "90b-vision")):
        return 0.005 / 10_000.0, "transaction"

    pricing = pricing_for_model(model_id, usage.get("input", 0))
    if pricing is None:
        return None, "unknown"
    input_cost = (usage.get("input", 0) / 1_000_000.0) * pricing.input_per_million
    output_cost = (usage.get("output", 0) / 1_000_000.0) * pricing.output_per_million
    return input_cost + output_cost, "token"


def emit_usage_observability(
    *,
    mode: str,
    model_id: str | None,
    session_id: str | None,
    thread_id: str | None,
    usage: dict[str, int] | None,
) -> tuple[dict[str, int] | None, float | None]:
    if usage is None:
        return None, None

    cost_usd, pricing_basis = estimate_cost_usd(model_id, usage)
    attributes = {
        "event_type": "llm_usage",
        "mode": mode,
        "model_id": model_id or "unknown",
        "session_id": session_id or "unknown",
        "thread_id": thread_id or "unknown",
        "input_tokens": usage.get("input", 0),
        "output_tokens": usage.get("output", 0),
        "total_tokens": usage.get("total", 0),
        "cost_usd": cost_usd or 0.0,
        "pricing_basis": pricing_basis,
    }
    logger.info(
        "llm_usage mode=%s model_id=%s session_id=%s thread_id=%s input_tokens=%d output_tokens=%d "
        "total_tokens=%d cost_usd=%.8f pricing_basis=%s",
        mode,
        model_id or "unknown",
        session_id or "unknown",
        thread_id or "unknown",
        usage.get("input", 0),
        usage.get("output", 0),
        usage.get("total", 0),
        cost_usd or 0.0,
        pricing_basis,
        extra={"otel_attributes": attributes},
    )
    return usage, cost_usd


__all__ = [
    "TokenPricing",
    "emit_usage_observability",
    "estimate_cost_usd",
    "extract_usage",
    "normalize_usage",
    "pricing_for_model",
    "to_int",
]

"""
Context window monitoring for LLM calls (inspired by Oracle AI Developer Hub notebook:
memory_context_engineering_agents.ipynb).

Estimates token usage of the prompt (system + chat history + user message) and
compares to model limits so we can log warnings or trim when approaching capacity.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based token estimation")

# Default context limit when model is unknown (conservative)
DEFAULT_MAX_TOKENS = 128_000

# Approximate context limits (input) for models used by the app (OCI/OpenAI/xAI/Google)
MODEL_TOKEN_LIMITS: dict[str, int] = {
    "meta.llama-3.3-70b-instruct": 128_000,
    "meta.llama-4-maverick-17b-128e-instruct-fp8": 128_000,
    "openai.gpt-4.1": 128_000,
    "openai.gpt-4o": 128_000,
    "openai.gpt-5": 256_000,
    "openai.gpt-4": 8_192,
    "openai.gpt-3.5-turbo": 16_385,
    "xai.grok-3": 128_000,
    "xai.grok-4": 128_000,
    "xai.grok-4-fast-reasoning": 128_000,
    "xai.grok-code-fast-1": 128_000,
    "google.gemini-2.5-pro": 128_000,
    "cohere.command-a-03-2025": 128_000,
}


def estimate_tokens(text: str, model_id: str | None = None) -> int:
    """
    Estimate token count using tiktoken when available, otherwise fallback to character length (~4 chars per token for English).
    """
    if not text:
        return 0

    if TIKTOKEN_AVAILABLE and model_id:
        try:
            # Map model IDs to tiktoken encodings
            model_to_encoding = {
                "gpt-4": "cl100k_base",
                "gpt-4o": "o200k_base",
                "gpt-4o-mini": "o200k_base",
                "gpt-5": "o200k_base",
                "text-embedding-ada-002": "cl100k_base",
                "text-embedding-3-small": "cl100k_base",
                "text-embedding-3-large": "cl100k_base",
            }

            # Try exact model match first
            encoding_name = model_to_encoding.get(model_id)
            if not encoding_name:
                # Try to find a matching encoding by prefix
                for model_prefix, enc in model_to_encoding.items():
                    if model_id.startswith(model_prefix):
                        encoding_name = enc
                        break

            if encoding_name:
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(text))
            else:
                # Fallback: try to get encoding for model directly
                try:
                    encoding = tiktoken.encoding_for_model(model_id)
                    return len(encoding.encode(text))
                except KeyError:
                    pass
        except Exception as e:
            logger.debug(f"Failed to use tiktoken for model {model_id}: {e}")

    # Fallback to character-based estimation
    return max(0, len(text) // 4)


def messages_to_text(messages: list[Any]) -> str:
    """
    Serialize LangChain-style messages to a single string for token estimation.
    Each message contributes role and content fields.
    """
    parts: list[str] = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", "message")
        content = getattr(m, "content", None) or ""
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"{role}: [non-string content]")
    return "\n".join(parts)


def calculate_context_usage(
    context_text: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Compute context window usage (tokens, max, percent).

    Args:
        context_text: Full prompt text (or serialized messages).
        model_id: Model identifier for lookup in MODEL_TOKEN_LIMITS.

    Returns:
        Dict with keys: tokens (int), max (int), percent (float), model_id (str).
    """
    tokens = estimate_tokens(context_text, model_id)
    max_tokens = MODEL_TOKEN_LIMITS.get(model_id or "", DEFAULT_MAX_TOKENS)
    percent = (tokens / max_tokens) * 100.0 if max_tokens > 0 else 0.0
    return {
        "tokens": tokens,
        "max": max_tokens,
        "percent": round(percent, 1),
        "model_id": model_id or "unknown",
    }


def log_context_usage(usage: dict[str, Any], threshold_percent: float = 80.0) -> None:
    """
    Log context usage; emit warning when above threshold (e.g. 80%).
    """
    pct = usage.get("percent", 0)
    tokens = usage.get("tokens", 0)
    max_tok = usage.get("max", 0)
    model = usage.get("model_id", "unknown")
    if pct >= threshold_percent:
        logger.warning(
            "Context window usage high: %.1f%% (%s tokens / %s max) for model %s",
            pct,
            tokens,
            max_tok,
            model,
        )
    else:
        logger.info(
            "Context window: %.1f%% (%s / %s tokens) model=%s",
            pct,
            tokens,
            max_tok,
            model,
        )

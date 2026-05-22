"""Token-estimation helpers for tracing LLM prompts and completions."""

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

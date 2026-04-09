"""AI SDK UI Message Stream helpers.

This module centralizes the SSE framing and response headers used by the
Next.js AI SDK UI Message Stream Protocol.

Routers import from here to keep streaming logic consistent across endpoints.
"""

from __future__ import annotations

import json
from typing import Final

MEDIA_TYPE_SSE: Final[str] = "text/event-stream"
AI_SDK_UI_MESSAGE_STREAM_HEADER_VALUE: Final[str] = "v1"

# Canonical headers for AI SDK UI message stream
AI_SDK_RESPONSE_HEADERS: Final[dict[str, str]] = {
    "content-type": MEDIA_TYPE_SSE,
    "cache-control": "no-cache",
    "x-vercel-ai-ui-message-stream": AI_SDK_UI_MESSAGE_STREAM_HEADER_VALUE,
    "x-accel-buffering": "no",
    "connection": "keep-alive",
}

# Stream terminator payload used by protocol
DONE_MARKER: Final[str] = "[DONE]"
DONE_FRAME: Final[str] = f"data: {DONE_MARKER}\n\n"


def ai_sdk_sse_frame(obj: object) -> str:
    """Format a JSON-serializable object as an AI SDK SSE frame.

    Returns a string like: "data: {<json>}\n\n"
    """
    return f"data: {json.dumps(obj)}\n\n"


def ai_sdk_error_event(message: str) -> dict[str, str]:
    """Build a generic, safe error event payload.

    Note: Do not leak raw exception details to clients. Provide a user-safe summary.
    """
    return {"type": "error", "errorText": message}

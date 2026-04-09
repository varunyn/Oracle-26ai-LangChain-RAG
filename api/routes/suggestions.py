"""Follow-up suggestions endpoint: generates short follow-up questions from the last assistant message."""

import json
import logging
import re
from typing import Any

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.rag_agent.infrastructure.oci_models import get_llm

router = APIRouter()
logger = logging.getLogger(__name__)

FOLLOW_UP_SYSTEM = """You suggest follow-up questions. Given an assistant's message, output 3 to 5 short follow-up questions a user might ask next.
Return only a JSON array of strings. No markdown, no explanation. Example: ["First question?","Second question?"]"""


class SuggestionsRequest(BaseModel):
    """Request body for POST /api/suggestions."""

    last_message: str = Field(..., description="Last assistant message text to base suggestions on")
    model: str | None = Field(default=None, description="Model ID; uses default if omitted")


class SuggestionsResponse(BaseModel):
    """Response for POST /api/suggestions."""

    suggestions: list[str] = Field(default_factory=list, description="Follow-up question strings")


async def _generate_suggestions_async(last_message: str, model_id: str | None) -> list[str]:
    llm = get_llm(
        model_id=model_id,
        temperature=0.3,
        max_tokens=300,
    )
    messages = [
        SystemMessage(content=FOLLOW_UP_SYSTEM),
        HumanMessage(content=last_message[:4000]),
    ]
    msg = await llm.ainvoke(messages)
    text = (getattr(msg, "content", None) or "").strip()
    raw = re.sub(r"^```json?\s*|\s*```$", "", text).strip()
    suggestions: list[str] = []
    try:
        parsed: Any = json.loads(raw)
        if isinstance(parsed, list):
            suggestions = [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:6]
    except (json.JSONDecodeError, TypeError):
        pass
    return suggestions


@router.post("/api/suggestions", response_model=SuggestionsResponse)
async def post_suggestions(request: SuggestionsRequest) -> SuggestionsResponse:
    """Generate 3–6 follow-up question suggestions from the last assistant message."""
    if not request.last_message.strip():
        return SuggestionsResponse(suggestions=[])
    try:
        suggestions = await _generate_suggestions_async(
            request.last_message.strip(),
            request.model,
        )
        return SuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        logger.exception("Suggestions generation failed: %s", e)
        return SuggestionsResponse(suggestions=[])

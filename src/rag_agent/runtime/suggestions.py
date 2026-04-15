"""Follow-up suggestions endpoint for the runtime API surface."""

import asyncio
import json
import logging
import re
from typing import Any

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import AliasChoices, BaseModel, Field

from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks

router = APIRouter(tags=["suggestions"])
logger = logging.getLogger(__name__)

FOLLOW_UP_SYSTEM = """You generate follow-up user questions for the CURRENT conversation only.
Output exactly 3 to 5 concise questions as a JSON array of strings.

Rules:
- Keep questions tightly grounded in the latest user question and assistant answer.
- Do not change domain/topic. No generic brainstorming.
- Each suggestion must be <= 12 words and end with "?".
- Avoid duplicates and near-duplicates.
- Return only JSON array. No markdown, no explanation.

Example:
["Can you show the exact steps in Visual Builder?","What prerequisites are required first?"]"""

RAW_TOOL_CALL_PATTERN = re.compile(r"^[\w\.]+\s*\([^)]*\)$")
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
GENERIC_SUGGESTION_PATTERN = re.compile(
    r"^(can you tell me more|what else|anything else|more details|explain more)\??$",
    re.IGNORECASE,
)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "about",
    "what",
    "when",
    "where",
    "which",
    "how",
    "can",
    "could",
    "would",
    "should",
    "into",
    "then",
    "than",
    "have",
    "has",
    "had",
    "you",
    "are",
    "was",
    "were",
    "they",
    "them",
    "their",
    "there",
    "here",
    "just",
    "also",
    "only",
    "using",
    "use",
    "used",
    "need",
    "show",
    "give",
    "make",
    "create",
}
EQUATION_HINT_PATTERN = re.compile(r"[=+\-*/^()]|\d")


class SuggestionsRequest(BaseModel):
    """Request body for POST /api/suggestions."""

    last_message: str = Field(
        ...,
        validation_alias=AliasChoices("last_message", "lastMessage"),
        description="Last assistant message text to base suggestions on",
    )
    last_user_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices("last_user_message", "lastUserMessage"),
        description="Latest user question to keep suggestions on-topic",
    )
    model: str | None = Field(default=None, description="Model ID; uses default if omitted")


class SuggestionsResponse(BaseModel):
    """Response for POST /api/suggestions."""

    suggestions: list[str] = Field(default_factory=list, description="Follow-up question strings")


def _looks_like_raw_tool_call(text: str) -> bool:
    candidate = text.strip()
    return bool(candidate) and len(candidate) <= 200 and bool(RAW_TOOL_CALL_PATTERN.fullmatch(candidate))


def _extract_keywords(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_PATTERN.findall(text)
        if token and token.lower() not in STOPWORDS
    }


def _normalize_question(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip())
    value = value.rstrip(".!")
    if value and not value.endswith("?"):
        value = f"{value}?"
    return value


def _filter_suggestions(
    *,
    suggestions: list[str],
    last_message: str,
    last_user_message: str | None,
) -> list[str]:
    topic_keywords = _extract_keywords(f"{last_user_message or ''} {last_message}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in suggestions:
        candidate = _normalize_question(raw)
        if not candidate:
            continue
        if GENERIC_SUGGESTION_PATTERN.fullmatch(candidate.strip().lower()):
            continue
        key = candidate.lower()
        if key in seen:
            continue
        suggestion_keywords = _extract_keywords(candidate)
        enforce_topic_overlap = bool(last_user_message and last_user_message.strip())
        if (
            enforce_topic_overlap
            and topic_keywords
            and suggestion_keywords
            and topic_keywords.isdisjoint(suggestion_keywords)
        ):
            continue
        seen.add(key)
        cleaned.append(candidate)
        if len(cleaned) >= 5:
            break
    return cleaned


def _fallback_suggestions(last_user_message: str | None, last_message: str) -> list[str]:
    user_text = (last_user_message or "").strip()
    if user_text:
        base = _normalize_question(user_text)
        return [
            f"{base[:-1]} with step-by-step details?",
            "Can you verify the final answer quickly?",
            "Can you show one similar example?",
        ]

    if EQUATION_HINT_PATTERN.search(last_message or ""):
        return [
            "Can you show each algebra step?",
            "Can you verify by substitution?",
            "Can you solve a similar equation?",
        ]

    topic_keywords = sorted(_extract_keywords(last_message))[:2]
    if topic_keywords:
        topic = " and ".join(topic_keywords)
        return [
            f"Can you explain {topic} step-by-step?",
            f"What should I do first for {topic}?",
            f"Can you give one practical {topic} example?",
        ]

    return [
        "Can you show this step-by-step?",
        "Can you verify the final answer quickly?",
        "Can you give one practical example?",
    ]


async def _generate_suggestions_async(
    *,
    last_message: str,
    last_user_message: str | None,
    model_id: str | None,
) -> list[str]:
    llm = get_llm(
        model_id=model_id,
        temperature=0.2,
        max_tokens=300,
    )
    user_context = (last_user_message or "").strip()
    prompt_payload = (
        f"Latest user question:\n{user_context[:2000] or '(none)'}\n\n"
        f"Latest assistant answer:\n{last_message[:4000]}"
    )
    messages = [
        SystemMessage(content=FOLLOW_UP_SYSTEM),
        HumanMessage(content=prompt_payload),
    ]
    run_config: dict[str, object] = {"configurable": {"mode": "suggestions", "model_id": model_id or ""}}
    add_langfuse_callbacks(run_config, session_id=None, user_id=None)

    def _invoke() -> object:
        try:
            return llm.invoke(messages, config=run_config)
        except TypeError:
            return llm.invoke(messages)

    msg = await asyncio.to_thread(_invoke)
    text = (getattr(msg, "content", None) or "").strip()
    raw = re.sub(r"^```json?\s*|\s*```$", "", text).strip()
    suggestions: list[str] = []
    try:
        parsed: Any = json.loads(raw)
        if isinstance(parsed, list):
            suggestions = [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:6]
    except (json.JSONDecodeError, TypeError):
        pass
    filtered = _filter_suggestions(
        suggestions=suggestions,
        last_message=last_message,
        last_user_message=last_user_message,
    )
    return filtered if filtered else _fallback_suggestions(last_user_message, last_message)


@router.post("/api/suggestions", response_model=SuggestionsResponse)
async def post_suggestions(request: SuggestionsRequest) -> SuggestionsResponse:
    """Generate 3-6 follow-up question suggestions from the last assistant message."""
    if not request.last_message.strip():
        return SuggestionsResponse(suggestions=[])
    if _looks_like_raw_tool_call(request.last_message):
        return SuggestionsResponse(suggestions=[])
    try:
        suggestions = await _generate_suggestions_async(
            last_message=request.last_message.strip(),
            last_user_message=(request.last_user_message or "").strip() or None,
            model_id=request.model,
        )
        return SuggestionsResponse(suggestions=suggestions)
    except Exception as e:  # noqa: BLE001
        logger.exception("Suggestions generation failed: %s", e)
        return SuggestionsResponse(suggestions=[])

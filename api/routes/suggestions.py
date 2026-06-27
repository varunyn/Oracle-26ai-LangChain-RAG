"""Follow-up suggestions endpoint for the runtime API surface."""

import asyncio
import logging
from typing import Any, cast

from fastapi import APIRouter
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import AliasChoices, BaseModel, Field, field_validator

from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks, start_langfuse_chat_trace

router = APIRouter(tags=["suggestions"])
logger = logging.getLogger(__name__)

FOLLOW_UP_SYSTEM = """You generate follow-up user questions for the CURRENT conversation only.
Output exactly 3 to 5 concise questions as a JSON array of strings.

Rules:
- Keep questions tightly grounded in the latest user question and assistant answer.
- Return an empty list if the assistant answer is empty, just internal tool syntax, or not enough context.
- Do not change domain/topic. No generic brainstorming.
- Each suggestion must be <= 12 words and end with "?".
- Avoid duplicates and near-duplicates.

Example:
["Can you show the exact steps in Visual Builder?","What prerequisites are required first?"]"""


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


class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up suggestions."""

    suggestions: list[str] = Field(
        default_factory=list,
        description="Three to five concise follow-up questions, or an empty list.",
    )

    @field_validator("suggestions", mode="before")
    @classmethod
    def _coerce_suggestions(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @field_validator("suggestions")
    @classmethod
    def _normalize_suggestions(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            question = " ".join(raw.strip().split()).rstrip(".!")
            if not question:
                continue
            if not question.endswith("?"):
                question = f"{question}?"
            if len(question[:-1].split()) > 12:
                continue
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(question)
            if len(normalized) >= 5:
                break
        return normalized


def _extract_structured_suggestions(result: object) -> list[str]:
    if isinstance(result, dict):
        structured = result.get("structured_response")
    else:
        structured = getattr(result, "structured_response", None)

    if isinstance(structured, FollowUpSuggestions):
        return structured.suggestions
    if isinstance(structured, dict):
        return FollowUpSuggestions.model_validate(structured).suggestions
    return []


def _should_retry_with_default_model(exc: BaseException, *, model_id: str | None) -> bool:
    if not model_id:
        return False
    if not isinstance(exc, TypeError):
        return False
    return "Unrecognized keyword arguments: strict" in str(exc)


async def _generate_suggestions_async(
    *,
    last_message: str,
    last_user_message: str | None,
    model_id: str | None,
) -> list[str]:
    user_context = (last_user_message or "").strip()
    prompt_payload = (
        f"Latest user question:\n{user_context[:2000] or '(none)'}\n\n"
        f"Latest assistant answer:\n{last_message[:4000]}"
    )
    trace_tags = [
        tag
        for tag in (
            "suggestions",
            f"model:{model_id}" if model_id else None,
        )
        if tag is not None
    ]
    for attempt_model_id in (model_id, None):
        llm = get_llm(
            model_id=attempt_model_id,
            temperature=0.2,
            max_tokens=300,
        )
        run_config: dict[str, object] = {
            "configurable": {"mode": "suggestions", "model_id": attempt_model_id or ""}
        }
        with start_langfuse_chat_trace(
            enabled=True,
            mode="suggestions",
            model_id=attempt_model_id,
            session_id=None,
            thread_id=None,
            input_payload={
                "last_user_message": user_context[:2000] or None,
                "last_message": last_message[:4000],
            },
            trace_name="suggestions",
            tags=trace_tags,
        ) as langfuse_trace:
            add_langfuse_callbacks(
                run_config,
                session_id=None,
                user_id=None,
                trace_context=langfuse_trace.trace_context,
                trace_name="suggestions",
                tags=trace_tags,
            )

            def _invoke() -> object:
                agent = create_agent(
                    model=llm,
                    tools=[],
                    system_prompt=FOLLOW_UP_SYSTEM,
                    response_format=FollowUpSuggestions,
                )
                return agent.invoke(
                    cast(Any, {"messages": [HumanMessage(content=prompt_payload)]}),
                    config=cast(Any, run_config),
                )

            try:
                result = await asyncio.to_thread(_invoke)
            except Exception as exc:
                if _should_retry_with_default_model(exc, model_id=attempt_model_id):
                    logger.info(
                        "Suggestions model %s rejected structured output strict mode; retrying with default model",
                        attempt_model_id,
                    )
                    continue
                raise
            suggestions = _extract_structured_suggestions(result)
            langfuse_trace.update_output({"suggestion_count": len(suggestions)})
            return suggestions
    return []


@router.post("/api/suggestions", response_model=SuggestionsResponse)
async def post_suggestions(request: SuggestionsRequest) -> SuggestionsResponse:
    """Generate 3-6 follow-up question suggestions from the last assistant message."""
    if not request.last_message.strip():
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

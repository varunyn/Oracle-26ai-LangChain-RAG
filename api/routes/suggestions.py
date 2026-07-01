"""Follow-up suggestions endpoint for the runtime API surface."""

import asyncio
import logging
from typing import Any, cast

from fastapi import APIRouter
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import AliasChoices, BaseModel, Field, field_validator

from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks, start_langfuse_chat_trace
from src.rag_agent.utils.logging_config import REQUEST_ID_CTX

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
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
        description="Chat thread ID used to group suggestions in Langfuse",
    )


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


def _has_length_finish_reason(value: object) -> bool:
    """Return whether a provider response contains a length finish reason."""
    pending = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, dict):
            if current.get("finish_reason") == "length":
                return True
            pending.extend(current.values())
            continue
        for attribute in ("response_metadata", "additional_kwargs"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)
        if isinstance(current, (list, tuple)):
            pending.extend(current)
    return False


async def _generate_suggestions_async(
    *,
    last_message: str,
    last_user_message: str | None,
    model_id: str | None,
    thread_id: str | None,
) -> list[str]:
    user_context = (last_user_message or "").strip()
    normalized_thread_id = (thread_id or "").strip() or None
    request_id = REQUEST_ID_CTX.get()
    normalized_request_id = request_id if request_id and request_id != "-" else None
    prompt_payload = (
        f"Latest user question:\n{user_context[:2000] or '(none)'}\n\n"
        f"Latest assistant answer:\n{last_message[:4000]}"
    )
    trace_tags = [
        tag
        for tag in (
            "feature:suggestions",
            "mode:suggestions",
            f"model:{model_id}" if model_id else None,
        )
        if tag is not None
    ]
    llm = get_llm(
        model_id=model_id,
        temperature=0.2,
        max_tokens=128,
    )
    run_config: dict[str, object] = {
        "configurable": {"mode": "suggestions", "model_id": model_id or ""},
        "metadata": {
            key: value
            for key, value in {
                "request_id": normalized_request_id,
                "thread_id": normalized_thread_id,
                "mode": "suggestions",
                "model_id": model_id,
            }.items()
            if value
        },
    }
    with start_langfuse_chat_trace(
        enabled=True,
        mode="suggestions",
        model_id=model_id,
        session_id=normalized_thread_id,
        thread_id=normalized_thread_id,
        request_id=normalized_request_id,
        metadata={
            key: value
            for key, value in {
                "request_id": normalized_request_id,
                "thread_id": normalized_thread_id,
            }.items()
            if value
        },
        input_payload={
            "last_user_message": user_context[:2000] or None,
            "last_message": last_message[:4000],
        },
        trace_name="suggestions.generate",
        tags=trace_tags,
    ) as langfuse_trace:
        add_langfuse_callbacks(
            run_config,
            session_id=normalized_thread_id,
            user_id=None,
            request_id=normalized_request_id,
            trace_context=langfuse_trace.trace_context,
            trace_name="suggestions.generate",
            tags=trace_tags,
        )

        def _invoke() -> object:
            agent = create_agent(
                model=llm,
                tools=[],
                system_prompt=FOLLOW_UP_SYSTEM,
                response_format=ToolStrategy(
                    FollowUpSuggestions,
                    tool_message_content="Returning follow-up suggestions.",
                ),
            )
            return agent.invoke(
                cast(Any, {"messages": [HumanMessage(content=prompt_payload)]}),
                config=cast(Any, run_config),
            )

        try:
            result = await asyncio.to_thread(_invoke)
        except Exception:
            langfuse_trace.update_output({"suggestion_count": 0, "outcome": "error"})
            langfuse_trace.update_metadata({"suggestion_count": "0", "outcome": "error"})
            update_outcome = getattr(langfuse_trace, "update_outcome", None)
            if callable(update_outcome):
                update_outcome("error", error_type="suggestions_generation")
            raise
        suggestions = _extract_structured_suggestions(result)
        outcome = (
            "truncated"
            if _has_length_finish_reason(result)
            else "success"
            if suggestions
            else "empty"
        )
        langfuse_trace.update_output(
            {"suggestion_count": len(suggestions), "outcome": outcome}
        )
        langfuse_trace.update_metadata(
            {"suggestion_count": str(len(suggestions)), "outcome": outcome}
        )
        update_outcome = getattr(langfuse_trace, "update_outcome", None)
        if callable(update_outcome):
            update_outcome(outcome)
        return suggestions


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
            thread_id=request.thread_id,
        )
        return SuggestionsResponse(suggestions=suggestions)
    except Exception as e:  # noqa: BLE001
        logger.exception("Suggestions generation failed: %s", e)
        return SuggestionsResponse(suggestions=[])

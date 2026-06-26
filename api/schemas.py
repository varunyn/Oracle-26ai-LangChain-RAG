"""Pydantic request/response models for the RAG Agent API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if normalized == "human":
            return "user"
        if normalized == "ai":
            return "assistant"
        return normalized


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Assistant answer")
    feedback: int = Field(..., description="Star rating 1-5")
    trace_id: str | None = Field(default=None, description="Langfuse trace id for this answer")

    @field_validator("feedback")
    @classmethod
    def feedback_range(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("feedback must be between 1 and 5")
        return v

"""Pydantic request/response models for the RAG Agent API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @model_validator(mode="before")
    @classmethod
    def _derive_content_from_parts(cls, data: object) -> object:
        """Allow AI SDK-style messages that include `parts` but omit `content`.
        If `content` is missing or empty and `parts` exists, extract text from any
        part objects where {"type": "text", "text": "..."} and concatenate.
        This is a pre-validation transform and does not change the OpenAPI schema.
        """
        if isinstance(data, dict):
            content = data.get("content")
            # Only derive when content is missing/empty and parts is present
            if (content is None or (isinstance(content, str) and content.strip() == "")) and (
                "parts" in data
            ):
                parts = data.get("parts")
                texts: list[str] = []
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") == "text":
                            t = p.get("text")
                            if isinstance(t, str):
                                texts.append(t)
                # Deterministic: set to concatenated text (possibly empty string)
                derived = "".join(texts)
                new_data = {**data}
                new_data["content"] = derived
                return new_data
        return data

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


class Citation(BaseModel):
    source: str
    page: str | None = None


class RerankerDoc(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    content: str
    standalone_question: str | None = None
    citations: list[Citation] | None = None
    reranker_docs: list[RerankerDoc] | None = None
    context_usage: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Assistant answer")
    feedback: int = Field(..., description="Star rating 1-5")

    @field_validator("feedback")
    @classmethod
    def feedback_range(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("feedback must be between 1 and 5")
        return v

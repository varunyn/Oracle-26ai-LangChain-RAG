"""Pydantic request/response models for the RAG Agent API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class InvokeRequest(BaseModel):
    user_input: str


class ChatMessage(BaseModel):
    role: str
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


class Citation(BaseModel):
    source: str
    page: str | None = None


class RerankerDoc(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionsRequest(BaseModel):
    model: str | None = Field(default=None, description="Model ID")
    messages: list[ChatMessage] = Field(
        ..., description="Conversation messages (OpenAI-compatible format)"
    )
    stream: bool = Field(default=False, description="If true, return SSE stream")
    thread_id: str | None = Field(
        default=None, description="Conversation thread ID for checkpointer memory"
    )
    session_id: str | None = Field(
        default=None,
        description="Browser/session ID for grouping traces (new per tab load or refresh)",
    )
    collection_name: str | None = Field(
        default=None, description="Vector store collection/table name"
    )
    enable_reranker: bool | None = Field(default=None, description="Enable reranker step")
    enable_tracing: bool | None = Field(default=None, description="Enable tracing")
    mode: str | None = Field(
        default=None,
        description="Flow mode: rag | mcp | mixed | direct; default rag for backward compat",
    )
    mcp_server_keys: list[str] | None = Field(
        default=None,
        description="MCP server keys from MCP_SERVERS_CONFIG to load tools",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_delta_only(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if not any(message.role == "user" for message in v):
            raise ValueError("messages must contain at least one user message")
        last_message = v[-1]
        if last_message.role != "user":
            raise ValueError("last message role must be 'user'")
        return v


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


class McpChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Conversation messages (OpenAI format)")
    mcp_url: str | None = Field(
        default=None,
        description="MCP server URL (e.g. http://host:port/mcp/); uses default if not set)",
    )
    stream: bool = Field(default=True, description="If true, return SSE stream")

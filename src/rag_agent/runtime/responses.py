"""Shared response shaping for chat/run APIs."""

from __future__ import annotations

import time
from typing import Any, cast

from api.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    Citation,
    RerankerDoc,
)
from api.settings import get_settings
from src.rag_agent.core.citations import normalize_citations


def to_citations(raw: list[dict[str, object]]) -> list[Citation]:
    citations: list[Citation] = []
    for item in normalize_citations(raw or []):
        citations.append(
            Citation(
                source=str(item.get("source", "")),
                page=cast(str | None, item.get("page")),
            )
        )
    return citations


def to_reranker_docs(raw: list[dict[str, object]]) -> list[RerankerDoc]:
    docs: list[RerankerDoc] = []
    for doc in raw or []:
        docs.append(
            RerankerDoc(
                page_content=str(doc.get("page_content", "")),
                metadata=cast(dict[str, Any], (doc.get("metadata") or {})),
            )
        )
    return docs


def chat_completion_response_json(
    content: str,
    model_id: str | None,
    completion_id: str,
    standalone_question: str | None = None,
    citations: list[dict[str, object]] | None = None,
    reranker_docs: list[dict[str, object]] | None = None,
    context_usage: dict[str, object] | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    prompt_tokens = int(usage.get("input", 0)) if isinstance(usage, dict) else 0
    completion_tokens = int(usage.get("output", 0)) if isinstance(usage, dict) else 0
    total_tokens = int(usage.get("total", 0)) if isinstance(usage, dict) else 0
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    response = ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=model_id or get_settings().LLM_MODEL_ID,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": content},
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        content=content,
        standalone_question=standalone_question,
        citations=to_citations(citations or []) if citations is not None else None,
        reranker_docs=to_reranker_docs(reranker_docs or []) if reranker_docs is not None else None,
        context_usage=context_usage,
    )
    return response.model_dump()

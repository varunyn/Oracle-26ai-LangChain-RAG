from __future__ import annotations

import re

from pydantic import BaseModel, Field

INLINE_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class RankedChunk(BaseModel):
    index: int = Field(description="Original position of the chunk in the input list (0-based).")
    score: float = Field(description="Relevance score; higher is more relevant.")


class RankedChunksResult(BaseModel):
    ranked_chunks: list[RankedChunk] = Field(
        default_factory=list,
        description="Chunks in order of relevance, most relevant first.",
    )


class StructuredRAGAnswer(BaseModel):
    markdown: str = Field(
        description=(
            "Complete final answer in markdown. Preserve the user's requested output format "
            "and place inline citation markers like [1], [2] where they support claims."
        )
    )
    valid_citation_ids: list[int] = Field(
        default_factory=list,
        description="Unique source numbers referenced by the markdown answer.",
    )


def extract_inline_citation_ids(markdown: str) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for match in INLINE_CITATION_PATTERN.finditer(markdown):
        citation_id = int(match.group(1))
        if citation_id not in seen:
            seen.add(citation_id)
            ordered.append(citation_id)
    return ordered


def validate_inline_citation_ids(citation_ids: list[int], max_source_id: int) -> list[int]:
    if max_source_id < 1:
        return []
    valid = set(range(1, max_source_id + 1))
    seen: set[int] = set()
    filtered: list[int] = []
    for citation_id in citation_ids:
        if citation_id in valid and citation_id not in seen:
            seen.add(citation_id)
            filtered.append(citation_id)
    return filtered


def validate_structured_markdown_answer(
    answer: StructuredRAGAnswer,
    max_source_id: int,
) -> tuple[str, list[int]]:
    markdown = answer.markdown.strip()
    inline_ids = validate_inline_citation_ids(extract_inline_citation_ids(markdown), max_source_id)
    declared_ids = validate_inline_citation_ids(answer.valid_citation_ids, max_source_id)

    if declared_ids:
        declared_set = set(declared_ids)
        validated_ids = [citation_id for citation_id in inline_ids if citation_id in declared_set]
    else:
        validated_ids = inline_ids

    return markdown, validated_ids

"""Retrieval rewrite node for mixed-mode V2."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.langgraph.state import MixedV2State, RetrievalIntent, SearchMode

logger = logging.getLogger(__name__)


class DeterministicIntentModel(BaseModel):
    standalone_question: str
    search_mode: SearchMode = "semantic"
    top_k: int | None = Field(default=None, ge=1)
    needs_llm_refinement: bool = False
    decision_reason: str = ""


def _normalize_user_request(text: str) -> str:
    return " ".join(text.split())


def _extract_top_k(text: str) -> int | None:
    match = re.search(r"\b(?:top|first|show|give me)\s+(\d{1,2})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None

_CONTEXTUAL_REFERENCE_PATTERN = re.compile(
    r"""
    \b(
        it|its|they|them|their|this|that|these|those|
        one|ones|former|latter
    )\b
    |
    ^\s*(
        and|also|instead|then|now|what\ about|how\ about
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VERSION_PATTERN = re.compile(
    r"""
    \b
    (?:
        (?P<prefix>v(?:ersion)?|ver|release|rel)\s*
        (?P<number_prefixed>\d+(?:\.\d+)*(?:\.x)?)
        |
        (?P<number_plain>\d+(?:\.\d+)+(?:\.x)?|\d+\.x)
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REWRITE_PROMPT = """
You rewrite the latest user request into a standalone retrieval question.

Goals:
- Preserve the user's exact intent.
- Resolve pronouns and elliptical references using recent conversation context.
- Do not add facts not supported by the conversation.
- If the latest user request is already standalone, return it unchanged.
- Keep the rewritten question concise and suitable for retrieval/search.

Return ONLY valid JSON with this exact schema:
{
  "standalone_question": string,
  "search_mode": "semantic" | "hybrid",
  "product_area": string | null,
  "doc_version": string | null,
  "language": string | null,
  "top_k": integer | null,
  "decision_reason": string
}

Notes:
- Only set product_area if the conversation explicitly identifies one.
- Only set doc_version if the conversation explicitly mentions one.
- Prefer search_mode="semantic" unless explicit metadata clearly supports hybrid.
- top_k should usually be null unless the user explicitly asks for a count or limit.

Latest user request:
{user_request}

Recent conversation:
{history}
""".strip()


class RewriteDecisionModel(BaseModel):
    standalone_question: str = Field(min_length=1)
    search_mode: Literal["semantic", "hybrid"] = "semantic"
    product_area: str | None = None
    doc_version: str | None = None
    language: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    decision_reason: str = ""


class RewriteForRetrieval:
    def __call__(self, state: MixedV2State) -> dict[str, object]:
        if state.get("intent") != "retrieve":
            return {}

        user_request = str(state.get("user_request") or "").strip()
        if not user_request:
            retrieval_intent = self._build_retrieval_intent(standalone_question="")
            return {
                "standalone_question": "",
                "retrieval_intent": retrieval_intent,
            }

        messages = list(state.get("messages") or [])
        standalone_question, rewrite_metadata = self._rewrite_question(user_request, messages)
        retrieval_intent = self._build_retrieval_intent(
            standalone_question=standalone_question,
            metadata=rewrite_metadata,
        )
        return {
            "standalone_question": standalone_question,
            "retrieval_intent": retrieval_intent,
        }

    def _rewrite_question(
        self,
        user_request: str,
        messages: Sequence[BaseMessage],
    ) -> tuple[str, dict[str, object]]:
        if not self._needs_contextual_rewrite(user_request):
            return user_request, self._fallback_metadata(user_request)

        history = self._history_summary(messages)
        prompt = _REWRITE_PROMPT.format(
            user_request=user_request,
            history=history,
        )

        try:
            llm = get_llm()
            parser = getattr(llm, "with_structured_output", None)

            if callable(parser):
                structured_llm = parser(RewriteDecisionModel)
                structured = structured_llm.invoke([HumanMessage(content=prompt)])
                if isinstance(structured, RewriteDecisionModel):
                    parsed = structured.model_dump(mode="python")
                elif isinstance(structured, Mapping):
                    parsed = dict(cast(Mapping[str, object], structured))
                else:
                    parsed = {}
            else:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw_content = getattr(response, "content", "")
                content = raw_content if isinstance(raw_content, str) else str(raw_content)
                parsed = self._parse_json_object(content)

            normalized = self._normalize_rewrite_decision(parsed, fallback=user_request)
            return normalized["standalone_question"], normalized
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval rewrite failed; using original request: %s", exc)
            return user_request, self._fallback_metadata(user_request)

    def _needs_contextual_rewrite(self, user_request: str) -> bool:
        return bool(_CONTEXTUAL_REFERENCE_PATTERN.search(user_request))

    def _history_summary(self, messages: Sequence[BaseMessage], max_items: int = 6) -> str:
        snippets: list[str] = []
        last_line = ""

        for message in list(messages)[-max_items:]:
            role = getattr(message, "type", "message")
            if role not in {"human", "ai"}:
                continue

            content = str(getattr(message, "content", "")).strip()
            if not content:
                continue

            compact = " ".join(content.split())
            if len(compact) > 300:
                compact = f"{compact[:300]}..."

            line = f"{role}: {compact}"
            if line == last_line:
                continue

            snippets.append(line)
            last_line = line

        return "\n".join(snippets) if snippets else "(no prior messages)"

    def _normalize_rewrite_decision(
        self,
        raw: object,
        *,
        fallback: str,
    ) -> dict[str, object]:
        if not isinstance(raw, Mapping):
            return self._fallback_metadata(fallback)

        standalone_question = str(raw.get("standalone_question") or "").strip() or fallback

        raw_search_mode = str(raw.get("search_mode") or "semantic").strip().lower()
        search_mode: SearchMode = "hybrid" if raw_search_mode == "hybrid" else "semantic"

        product_area = self._normalize_optional_string(raw.get("product_area"))
        doc_version = self._normalize_doc_version(raw.get("doc_version")) or self._extract_doc_version(
            standalone_question
        )
        language = self._normalize_optional_string(raw.get("language"))

        top_k_raw = raw.get("top_k")
        top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else None

        if product_area is None and doc_version is None and search_mode == "hybrid":
            search_mode = "semantic"

        return {
            "standalone_question": standalone_question,
            "search_mode": search_mode,
            "product_area": product_area,
            "doc_version": doc_version,
            "language": language,
            "top_k": top_k,
            "decision_reason": str(raw.get("decision_reason") or "").strip(),
        }

    def _build_retrieval_intent(
        self,
        standalone_question: str,
        metadata: Mapping[str, object] | None = None,
    ) -> RetrievalIntent:
        metadata = metadata or {}

        product_area = self._normalize_optional_string(metadata.get("product_area"))
        doc_version = self._normalize_doc_version(metadata.get("doc_version")) or self._extract_doc_version(
            standalone_question
        )
        language = self._normalize_optional_string(metadata.get("language"))

        top_k_raw = metadata.get("top_k")
        top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else None

        raw_search_mode = str(metadata.get("search_mode") or "semantic").strip().lower()
        search_mode: SearchMode = "hybrid" if raw_search_mode == "hybrid" else "semantic"

        metadata_filters: dict[str, object] | None = None
        compact_filters = {
            key: value
            for key, value in {
                "product_area": product_area,
                "doc_version": doc_version,
            }.items()
            if value is not None
        }
        if compact_filters:
            metadata_filters = compact_filters

        if metadata_filters is None and search_mode == "hybrid":
            search_mode = "semantic"

        return {
            "standalone_question": standalone_question,
            "search_mode": search_mode,
            "metadata_filters": metadata_filters,
            "product_area": product_area,
            "doc_version": doc_version,
            "language": language,
            "top_k": top_k,
        }

    def _fallback_metadata(self, standalone_question: str) -> dict[str, object]:
        return {
            "standalone_question": standalone_question,
            "search_mode": "semantic",
            "product_area": None,
            "doc_version": self._extract_doc_version(standalone_question),
            "language": None,
            "top_k": None,
            "decision_reason": "fallback to original request",
        }

    def _extract_doc_version(self, text: str) -> str | None:
        match = _VERSION_PATTERN.search(text)
        if not match:
            return None

        raw_value = match.group("number_prefixed") or match.group("number_plain")
        return self._normalize_doc_version(raw_value)

    def _normalize_doc_version(self, value: object) -> str | None:
        normalized = self._normalize_optional_string(value)
        if normalized is None:
            return None

        cleaned = normalized.strip().lower()

        if cleaned.startswith("version "):
            cleaned = cleaned[len("version ") :].strip()
        elif cleaned.startswith("ver "):
            cleaned = cleaned[len("ver ") :].strip()
        elif cleaned.startswith("release "):
            cleaned = cleaned[len("release ") :].strip()
        elif cleaned.startswith("rel "):
            cleaned = cleaned[len("rel ") :].strip()
        elif cleaned.startswith("v "):
            cleaned = cleaned[len("v ") :].strip()

        cleaned = cleaned.lstrip("v").strip()

        if not cleaned:
            return None

        if not re.fullmatch(r"\d+(?:\.\d+)*(?:\.x)?|\d+\.x", cleaned, flags=re.IGNORECASE):
            return None

        return f"v{cleaned}"

    def _normalize_optional_string(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _parse_json_object(self, content: str) -> dict[str, object]:
        stripped = content.strip()

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, Mapping):
                return dict(cast(Mapping[str, object], parsed))
        except json.JSONDecodeError:
            pass

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                inner = "\n".join(lines[1:-1]).strip()
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, Mapping):
                        return dict(cast(Mapping[str, object], parsed))
                except json.JSONDecodeError:
                    pass

        raise ValueError("Rewrite response did not contain valid JSON")

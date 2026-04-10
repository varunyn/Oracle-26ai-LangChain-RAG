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
        deterministic = self._deterministic_intent(user_request, messages)
        if not deterministic.needs_llm_refinement:
            return deterministic.standalone_question, self._fallback_metadata(deterministic)

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

            normalized = self._normalize_rewrite_decision(parsed, fallback=deterministic)
            return normalized["standalone_question"], normalized
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval rewrite failed; using original request: %s", exc)
            return deterministic.standalone_question, self._fallback_metadata(deterministic)

    def _deterministic_intent(
        self, user_request: str, messages: Sequence[BaseMessage]
    ) -> DeterministicIntentModel:
        _ = messages
        normalized = _normalize_user_request(user_request)
        top_k = _extract_top_k(normalized)
        normalized_lower = normalized.lower()
        has_exact_tokens = bool(
            re.search(r'("[^"]+"|`[^`]+`|--[a-z0-9-]+)', normalized, re.IGNORECASE)
        )
        has_codeish_tokens = bool(
            re.search(r"\b[a-z0-9_./:-]*[-_/.:][a-z0-9_./:-]*\b", normalized_lower)
        )
        needs_context = self._needs_contextual_rewrite(normalized)

        if has_exact_tokens and any(
            term in normalized_lower for term in ["what does", "flag", "parameter", "command"]
        ):
            search_mode: SearchMode = "keyword"
            reason = "exact-term query detected"
        elif has_exact_tokens or has_codeish_tokens:
            search_mode = "hybrid"
            reason = "mixed exact-term and semantic cues detected"
        else:
            search_mode = "semantic"
            reason = "broad conceptual query detected"

        return DeterministicIntentModel(
            standalone_question=normalized,
            search_mode=search_mode,
            top_k=top_k,
            needs_llm_refinement=needs_context,
            decision_reason=reason,
        )

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
        fallback: DeterministicIntentModel,
    ) -> dict[str, object]:
        if not isinstance(raw, Mapping):
            return self._fallback_metadata(fallback)

        standalone_question = str(raw.get("standalone_question") or "").strip() or fallback.standalone_question

        raw_search_mode = str(raw.get("search_mode") or fallback.search_mode).strip().lower()
        search_mode: SearchMode = (
            cast(SearchMode, raw_search_mode)
            if raw_search_mode in {"semantic", "hybrid", "keyword"}
            else fallback.search_mode
        )

        product_area = self._normalize_optional_string(raw.get("product_area"))
        doc_version = self._normalize_optional_string(raw.get("doc_version"))
        language = self._normalize_optional_string(raw.get("language"))

        top_k_raw = raw.get("top_k")
        top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else fallback.top_k

        return {
            "standalone_question": standalone_question,
            "search_mode": search_mode,
            "product_area": product_area,
            "doc_version": doc_version,
            "language": language,
            "top_k": top_k,
            "decision_reason": str(raw.get("decision_reason") or fallback.decision_reason).strip(),
        }

    def _build_retrieval_intent(
        self,
        standalone_question: str,
        metadata: Mapping[str, object] | None = None,
    ) -> RetrievalIntent:
        metadata = metadata or {}

        product_area = self._normalize_optional_string(metadata.get("product_area"))
        doc_version = self._normalize_optional_string(metadata.get("doc_version"))
        language = self._normalize_optional_string(metadata.get("language"))

        top_k_raw = metadata.get("top_k")
        top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else None

        raw_search_mode = str(metadata.get("search_mode") or "semantic").strip().lower()
        search_mode: SearchMode = (
            cast(SearchMode, raw_search_mode)
            if raw_search_mode in {"semantic", "hybrid", "keyword"}
            else "semantic"
        )

        metadata_filters = None

        return {
            "standalone_question": standalone_question,
            "search_mode": search_mode,
            "metadata_filters": metadata_filters,
            "product_area": product_area,
            "doc_version": doc_version,
            "language": language,
            "top_k": top_k,
        }

    def _fallback_metadata(self, deterministic: DeterministicIntentModel) -> dict[str, object]:
        return {
            "standalone_question": deterministic.standalone_question,
            "search_mode": deterministic.search_mode,
            "product_area": None,
            "doc_version": None,
            "language": None,
            "top_k": deterministic.top_k,
            "decision_reason": deterministic.decision_reason,
        }

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

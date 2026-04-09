"""Answer generation nodes (RAG-only structured answer + final draft merge).

Moved from src/rag_agent/answer_generator.py to this location as part of
langgraph nodes migration. Behavior preserved; only imports adjusted.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Protocol, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import override, runtime_checkable

from api.settings import get_settings

from ...agent_state import State
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.oci_models import get_llm
from ...prompts import ANSWER_STRUCTURED_PROMPT_TEMPLATE
from ...schemas import (
    StructuredRAGAnswer,
    validate_structured_markdown_answer,
)
from ...utils.context_window import (
    calculate_context_usage,
    log_context_usage,
    messages_to_text,
)


class StructuredLLM(Protocol):
    """Protocol for an LLM bound to structured output (e.g. with_structured_output)."""

    def invoke(
        self, messages: list[HumanMessage], *, config: RunnableConfig
    ) -> StructuredRAGAnswer | object: ...


@runtime_checkable
class SupportsStructuredOutput(Protocol):
    """Protocol for LLMs that support with_structured_output (e.g. OCI Chat, OpenAI)."""

    def with_structured_output(
        self, model: type[StructuredRAGAnswer], *, method: str = ...
    ) -> StructuredLLM: ...


logger = logging.getLogger(__name__)

STRUCTURED_FAILED_MESSAGE = "**I couldn't generate a cited answer. Please try again.**"
REMOVE_ALL_MESSAGES_ID = "__remove_all__"


class AnswerFromDocs(Runnable[State, dict[str, object]]):
    """
    RAG-only answer from reranker_docs (no MCP). Writes rag_answer, citations, context_usage.

    Uses only structured output (markdown + citation_ids); we validate citations
    without rewriting answer text so user-requested formatting is preserved.
    If structured output fails, returns a short failure message (no freeform fallback).
    """

    def build_context_for_llm(self, docs: list[object]) -> str:
        """Prefix chunks with [1], [2], ... for citations."""
        parts: list[str] = []
        for i, doc in enumerate(docs):
            num = i + 1
            if isinstance(doc, dict):
                doc_map = cast(dict[str, object], doc)
                content = str(doc_map.get("page_content", ""))
            else:
                content = str(getattr(doc, "page_content", str(doc)))
            parts.append(f"[{num}] {content}")
        return "\n\n".join(parts)

    def _try_structured_path(
        self,
        llm: object,
        the_question: str,
        context: str,
        chat_history_text: str,
        num_sources: int,
        run_config: RunnableConfig,
    ) -> str | None:
        """
        Try to get a structured markdown answer from the LLM and validate citations.
        Returns rag_answer string or None if we should fall back.
        """
        prompt = PromptTemplate.from_template(ANSWER_STRUCTURED_PROMPT_TEMPLATE)
        formatted = prompt.format(
            question=the_question,
            chat_history=chat_history_text,
            context=context,
            num_sources=num_sources,
        )
        messages = [HumanMessage(content=formatted)]

        try:
            if isinstance(llm, SupportsStructuredOutput):
                try:
                    structured_llm: StructuredLLM = llm.with_structured_output(
                        StructuredRAGAnswer,
                        method="json_schema",
                    )
                    result = structured_llm.invoke(messages, config=run_config)
                    if not isinstance(result, StructuredRAGAnswer):
                        logger.warning(
                            "RAG structured output wrong type: %s",
                            type(result).__name__,
                        )
                        return None
                    if not result.markdown.strip():
                        logger.warning(
                            "RAG structured output returned empty markdown num_sources=%d",
                            num_sources,
                        )
                        return None
                    markdown, citation_ids = validate_structured_markdown_answer(
                        result,
                        num_sources,
                    )
                    if citation_ids or "don't know the answer" in markdown.lower():
                        logger.info(
                            "RAG citations path=structured (json_schema) num_sources=%d citations=%d",
                            num_sources,
                            len(citation_ids),
                        )
                        return markdown
                    logger.warning(
                        "RAG structured output had no valid inline citations num_sources=%d",
                        num_sources,
                    )
                    return None
                except Exception as e:
                    logger.warning(
                        "RAG structured output failed (json_schema): %s",
                        e,
                        exc_info=False,
                    )
                    return None
            return None
        except Exception as e:
            logger.warning("Structured RAG answer parsing failed: %s", e)
            return None

    @override
    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """
        Generate RAG answer from docs.
        Structured output only (citation-accurate). No freeform fallback.
        """

        log_node_start("AnswerFromDocs")
        t0 = time.perf_counter()
        run_config = config or {}
        model_id = run_config.get("configurable", {}).get("model_id") or get_settings().LLM_MODEL_ID

        the_question = input.get("user_request", "")
        docs = cast(list[object], input.get("reranker_docs") or [])
        context = self.build_context_for_llm(docs)
        if not (context and context.strip()):
            context = "No relevant documents were found for your query."
        history_messages = cast(list[object], input.get("messages", []))
        cached_history_text = str(input.get("history_text") or "").strip()
        chat_history_text = cached_history_text or messages_to_text(history_messages)
        citations = input.get("citations", [])

        num_sources = len(docs)
        if num_sources == 0:
            rag_answer: str = "**I don't know the answer.**"
        else:
            llm = get_llm(model_id=model_id)
            maybe_answer = self._try_structured_path(
                llm,
                the_question,
                context,
                chat_history_text,
                num_sources,
                run_config,
            )
            if maybe_answer is None:
                rag_answer = STRUCTURED_FAILED_MESSAGE
                logger.warning(
                    "RAG citations structured path failed, returning fallback message num_sources=%d",
                    num_sources,
                )
            else:
                rag_answer = maybe_answer

        # Context usage from prompt + final answer (for structured we don't have raw stream)
        messages_for_usage = [
            SystemMessage(content="You are an AI assistant."),
            HumanMessage(
                content=(
                    f"Question: {the_question}\n"
                    f"Chat history: {chat_history_text}\n"
                    f"Context: {context}"
                )
            ),
            HumanMessage(content=rag_answer),
        ]
        context_text = messages_to_text(messages_for_usage)
        context_usage = calculate_context_usage(context_text, model_id)
        log_context_usage(context_usage)

        duration_ms = (time.perf_counter() - t0) * 1000
        log_node_end("AnswerFromDocs", duration_ms=duration_ms, num_sources=num_sources)
        return {
            "rag_answer": rag_answer or "",
            "rag_context": context,
            "rag_has_citations": bool(re.search(r"\[\d+\]", rag_answer or "")),
            "citations": citations,
            "context_usage": context_usage,
        }

    @override
    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """Async entrypoint for LangGraph async compilation; runs sync invoke in thread."""
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)


class DraftAnswer(Runnable[State, dict[str, object]]):
    """
    Merges rag_answer, mcp_answer, or direct answer into final_answer.
    """

    @override
    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        return self._merge_answers(input, config)

    @override
    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        return self._merge_answers(input, config)

    @staticmethod
    def _bounded_messages_for_checkpoint(
        existing_messages: list[AnyMessage], final_answer: str, max_messages: int
    ) -> list[AnyMessage]:
        merged: list[AnyMessage] = [*existing_messages, AIMessage(content=final_answer)]
        if max_messages > 0 and len(merged) > max_messages:
            return merged[-max_messages:]
        return merged

    def _merge_answers(
        self, input: State, _config: RunnableConfig | None = None
    ) -> dict[str, object]:
        log_node_start("DraftAnswer")
        rag = (input.get("rag_answer") or "").strip()
        mcp = (input.get("mcp_answer") or "").strip()
        direct = (input.get("direct_answer") or "").strip()
        mode = (input.get("mode") or "").lower()
        route = (input.get("route") or "").lower()
        tools_used = input.get("mcp_tools_used") or []

        if direct:
            final = direct
        elif mode == "mcp":
            final = mcp
        elif mode == "rag":
            final = rag
        elif mode == "mixed":
            if tools_used and mcp:
                final = mcp
            elif tools_used and rag and "i don't know" not in rag.lower():
                final = rag
            elif tools_used:
                final = mcp or "The tool was used; see the result above."
            else:
                final = mcp or rag
        else:
            final = mcp or rag
        final = final or ""
        existing_messages: list[AnyMessage] = input.get("messages") or []
        max_messages = max(2, int(get_settings().MAX_MSGS_IN_HISTORY))
        bounded_messages = self._bounded_messages_for_checkpoint(
            existing_messages, final, max_messages
        )

        out: dict[str, object] = {
            "final_answer": final,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES_ID), *bounded_messages],
        }
        out["context_usage"] = input.get("context_usage")
        out["mcp_used"] = False if mode == "rag" else input.get("mcp_used", False)
        out["mcp_tools_used"] = [] if mode == "rag" else (input.get("mcp_tools_used") or [])
        tools_used_list = [] if mode == "rag" else tools_used

        if direct:
            answer_source = "direct"
        elif mode == "mcp":
            answer_source = "mcp"
        elif final == mcp and mcp:
            answer_source = "mcp"
        else:
            answer_source = "rag"

        out["citations"] = (
            input.get("citations", [])
            if answer_source != "mcp" and input.get("citations") is not None
            else []
        )
        answer_len = len(final)
        doc_count = len(input.get("reranker_docs") or input.get("retriever_docs") or [])
        attributes = {
            "event_type": "flow_trace",
            "route": route or "-",
            "mode": mode or "-",
            "tools_used": tools_used_list,
            "answer_source": answer_source,
            "answer_len": answer_len,
            "docs": doc_count,
        }
        logger.info(
            "flow_trace route=%s mode=%s tools_used=%s answer_source=%s answer_len=%d docs=%d",
            route or "-",
            mode or "-",
            tools_used_list,
            answer_source,
            answer_len,
            doc_count,
            extra={"otel_attributes": attributes},
        )
        log_node_end("DraftAnswer", next="END", answer_source=answer_source, answer_len=len(final))
        return out

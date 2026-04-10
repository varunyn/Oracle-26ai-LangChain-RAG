
"""RAG branch node for mixed-mode V2."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig

from api.settings import get_settings
from src.rag_agent.agent_state import State
from src.rag_agent.infrastructure.db_utils import get_pooled_connection
from src.rag_agent.infrastructure.oci_models import get_embedding_model, get_llm
from src.rag_agent.infrastructure.retrieval import normalize_search_mode, search_documents
from src.rag_agent.langgraph.retrieval_utils import (
    STRUCTURED_FAILED_MESSAGE,
    is_rag_answer_grounded,
    rerank_docs_v2,
)
from src.rag_agent.langgraph.state import CitationEntry, MixedV2State
from src.rag_agent.prompts import ANSWER_STRUCTURED_PROMPT_TEMPLATE
from src.rag_agent.schemas import StructuredRAGAnswer, validate_structured_markdown_answer
from src.rag_agent.utils.context_window import (
    calculate_context_usage,
    log_context_usage,
    messages_to_text,
)
from src.rag_agent.utils.utils import docs_serializable

logger = logging.getLogger(__name__)


class StructuredLLM(Protocol):
    def invoke(
        self, messages: list[HumanMessage], *, config: RunnableConfig
    ) -> StructuredRAGAnswer | object: ...


@runtime_checkable
class SupportsStructuredOutput(Protocol):
    def with_structured_output(
        self, model: type[StructuredRAGAnswer], *, method: str = ...
    ) -> StructuredLLM: ...


class RunRAG:

    def __call__(self, state: MixedV2State) -> dict[str, object]:
        retrieval_intent = state.get("retrieval_intent")
        if not isinstance(retrieval_intent, Mapping):
            return {
                "rag_result": {
                    "status": "unavailable",
                    "answer": "Retrieval intent was not provided.",
                    "citations": [],
                    "docs_used": 0,
                    "quality_score": 0.0,
                },
                "reranker_docs": [],
                "context_usage": None,
                "last_status": "unavailable",
            }

        standalone_question = str(
            retrieval_intent.get("standalone_question")
            or state.get("standalone_question")
            or state.get("user_request")
            or ""
        ).strip()
        if not standalone_question:
            return {
                "rag_result": {
                    "status": "unavailable",
                    "answer": "Retrieval question was empty.",
                    "citations": [],
                    "docs_used": 0,
                    "quality_score": 0.0,
                },
                "reranker_docs": [],
                "context_usage": None,
                "last_status": "unavailable",
            }

        run_config = self._build_run_config(state, retrieval_intent)
        legacy_state = self._build_legacy_state(state, standalone_question)

        search_update = self._retrieve_docs_v2(legacy_state, config=run_config)
        rerank_input = cast(State, cast(object, {**legacy_state, **search_update}))
        rerank_update = self._rerank_docs_v2(rerank_input, config=run_config)
        answer_input = cast(State, cast(object, {**rerank_input, **rerank_update}))
        answer_update = self._answer_from_docs_v2(answer_input, config=run_config)

        rag_answer = str(answer_update.get("rag_answer") or "").strip()
        raw_citations = cast(list[object], answer_update.get("citations") or [])
        citations = [cast(CitationEntry, item) for item in raw_citations if isinstance(item, dict)]
        reranker_docs = cast(list[dict[str, object]], rerank_update.get("reranker_docs") or [])
        context_usage = cast(dict[str, object] | None, answer_update.get("context_usage"))
        docs_used = len(reranker_docs)
        rag_answer = self._normalize_v2_answer(rag_answer, citations=citations, docs_used=docs_used)
        rag_has_citations = bool(citations) and self._is_v2_grounded_answer(rag_answer, reranker_docs)

        if not rag_answer:
            status = "unavailable"
            quality_score = 0.0
        elif rag_has_citations:
            status = "success"
            quality_score = 0.95
        elif citations:
            status = "ungrounded"
            quality_score = 0.35
        else:
            status = "unavailable"
            quality_score = 0.1

        return {
            "standalone_question": search_update.get("standalone_question") or standalone_question,
            "reranker_docs": reranker_docs,
            "context_usage": context_usage,
            "citations": citations,
            "rag_result": {
                "status": status,
                "answer": rag_answer,
                "citations": citations,
                "docs_used": docs_used,
                "quality_score": quality_score,
            },
            "last_status": status,
        }

    def _retrieve_docs_v2(
        self,
        input: State,
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        run_config = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        collection_name = str(configurable.get("collection_name") or get_settings().DEFAULT_COLLECTION)
        embed_model_type = str(configurable.get("embed_model_type") or get_settings().EMBED_MODEL_TYPE)
        search_mode = normalize_search_mode(
            str(configurable.get("search_mode") or get_settings().RAG_SEARCH_MODE)
        )
        top_k = int(configurable.get("top_k") or get_settings().TOP_K)
        metadata_filters = configurable.get("metadata_filters")
        contextualized_question = str(input.get("standalone_question") or input.get("user_request") or "")
        if not contextualized_question:
            return {"retriever_docs": [], "standalone_question": "", "error": None}

        try:
            embed_model = get_embedding_model(embed_model_type)
            with get_pooled_connection() as conn:
                relevant_docs = search_documents(
                    conn=conn,
                    collection_name=collection_name,
                    embed_model=embed_model,
                    query=contextualized_question,
                    top_k=top_k,
                    search_mode=search_mode,
                    metadata_filters=cast(dict[str, object] | None, metadata_filters),
                )
            return {
                "retriever_docs": docs_serializable(relevant_docs),
                "standalone_question": contextualized_question,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in V2 retrieval: %s", exc)
            return {
                "retriever_docs": [],
                "standalone_question": contextualized_question,
                "error": str(exc),
            }

    def _rerank_docs_v2(
        self,
        input: State,
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        _ = config
        return rerank_docs_v2(input)

    def _build_run_config(
        self, state: MixedV2State, retrieval_intent: Mapping[str, object]
    ) -> RunnableConfig:
        search_mode_raw = str(retrieval_intent.get("search_mode") or get_settings().RAG_SEARCH_MODE).strip().lower()
        search_mode = {"semantic": "vector", "keyword": "text", "hybrid": "hybrid"}.get(search_mode_raw, "vector")
        top_k_raw = retrieval_intent.get("top_k")
        top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else get_settings().TOP_K
        metadata_filters = retrieval_intent.get("metadata_filters")
        configurable: dict[str, Any] = {
            "collection_name": state.get("collection_name") or get_settings().DEFAULT_COLLECTION,
            "search_mode": search_mode,
            "top_k": top_k,
            "metadata_filters": metadata_filters if isinstance(metadata_filters, Mapping) else None,
            "enable_reranker": True,
            "model_id": get_settings().LLM_MODEL_ID,
        }
        mode = state.get("mode")
        if mode:
            configurable["mode"] = mode
        return cast(RunnableConfig, cast(object, {"configurable": configurable}))

    def _build_legacy_state(self, state: MixedV2State, standalone_question: str) -> State:
        return cast(
            State,
            cast(
                object,
                {
                    "user_request": standalone_question,
                    "standalone_question": standalone_question,
                    "messages": list(state.get("messages") or []),
                    "history_text": "",
                    "retriever_docs": [],
                    "reranker_docs": [],
                    "citations": [],
                    "error": None,
                },
            ),
        )

    def _build_context_for_llm(self, docs: list[object]) -> str:
        parts: list[str] = []
        for index, doc in enumerate(docs, start=1):
            if isinstance(doc, dict):
                content = str(doc.get("page_content", ""))
            else:
                content = str(getattr(doc, "page_content", str(doc)))
            parts.append(f"[{index}] {content}")
        return "\n\n".join(parts)

    def _try_structured_path(
        self,
        llm: object,
        question: str,
        context: str,
        chat_history_text: str,
        num_sources: int,
        run_config: RunnableConfig,
    ) -> str | None:
        prompt = PromptTemplate.from_template(ANSWER_STRUCTURED_PROMPT_TEMPLATE)
        formatted = prompt.format(
            question=question,
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
                        logger.warning("V2 structured output wrong type: %s", type(result).__name__)
                        return None
                    if not result.markdown.strip():
                        logger.warning(
                            "V2 structured output returned empty markdown num_sources=%d",
                            num_sources,
                        )
                        return None
                    markdown, citation_ids = validate_structured_markdown_answer(result, num_sources)
                    if citation_ids or "don't know the answer" in markdown.lower():
                        return markdown
                    logger.warning(
                        "V2 structured output had no valid inline citations num_sources=%d",
                        num_sources,
                    )
                    return None
                except Exception as exc:
                    logger.warning("V2 structured output failed (json_schema): %s", exc, exc_info=False)
                    return None
            return None
        except Exception as exc:
            logger.warning("V2 structured answer parsing failed: %s", exc)
            return None

    def _answer_from_docs_v2(
        self,
        answer_input: State,
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        run_config = config or {}
        model_id = run_config.get("configurable", {}).get("model_id") or get_settings().LLM_MODEL_ID
        question = str(answer_input.get("user_request") or "")
        docs = cast(list[object], answer_input.get("reranker_docs") or [])
        context = self._build_context_for_llm(docs)
        if not (context and context.strip()):
            context = "No relevant documents were found for your query."
        history_messages = cast(list[object], answer_input.get("messages", []))
        cached_history_text = str(answer_input.get("history_text") or "").strip()
        chat_history_text = cached_history_text or messages_to_text(history_messages)
        citations = cast(list[object], answer_input.get("citations") or [])

        num_sources = len(docs)
        if num_sources == 0:
            rag_answer = "**I don't know the answer.**"
        else:
            llm = get_llm(model_id=model_id)
            maybe_answer = self._try_structured_path(
                llm,
                question,
                context,
                chat_history_text,
                num_sources,
                cast(RunnableConfig, run_config),
            )
            rag_answer = maybe_answer if maybe_answer is not None else STRUCTURED_FAILED_MESSAGE

        messages_for_usage = [
            SystemMessage(content="You are an AI assistant."),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Chat history: {chat_history_text}\n"
                    f"Context: {context}"
                )
            ),
            HumanMessage(content=rag_answer),
        ]
        context_text = messages_to_text(messages_for_usage)
        context_usage = calculate_context_usage(context_text, str(model_id))
        log_context_usage(context_usage)
        return {
            "rag_answer": rag_answer or "",
            "rag_context": context,
            "rag_has_citations": is_rag_answer_grounded(rag_answer or "", docs),
            "latest_answer": rag_answer or "",
            "citations": citations,
            "context_usage": context_usage,
        }

    def _normalize_v2_answer(
        self,
        rag_answer: str,
        *,
        citations: list[CitationEntry],
        docs_used: int,
    ) -> str:
        if rag_answer != STRUCTURED_FAILED_MESSAGE:
            return rag_answer
        if citations or docs_used > 0:
            return (
                "I found relevant documentation, but could not produce a fully grounded cited answer "
                "from the retrieved content."
            )
        return rag_answer

    def _is_v2_grounded_answer(
        self,
        rag_answer: str,
        reranker_docs: list[dict[str, object]],
    ) -> bool:
        return is_rag_answer_grounded(rag_answer, cast(list[object], reranker_docs))

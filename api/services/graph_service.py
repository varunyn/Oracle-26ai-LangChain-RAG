"""ChatRuntimeService: runtime boundary between FastAPI and OCI-backed chat execution."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool, create_retriever_tool

from src.rag_agent.core.citations import citations_from_documents
from src.rag_agent.infrastructure.db_utils import get_pooled_connection
from src.rag_agent.infrastructure.direct_mcp_tools import get_mcp_tools_async
from src.rag_agent.infrastructure.mcp_agent import get_mcp_answer_async
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config
from src.rag_agent.infrastructure.oci_models import get_embedding_model, get_llm, get_oracle_vs
from src.rag_agent.infrastructure.retrieval import search_documents
from src.rag_agent.prompts.runtime_agents import RAG_ANSWER_PROMPT_TEMPLATE
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TokenPricing:
    input_per_million: float
    output_per_million: float


def _build_run_config(
    *,
    thread_id: str | None,
    mcp_server_keys: list[str] | None,
) -> RunnableConfig | None:
    configurable: dict[str, object] = {}
    if thread_id:
        configurable["thread_id"] = thread_id
    if mcp_server_keys:
        configurable["mcp_server_keys"] = mcp_server_keys
    if not configurable:
        return None
    return cast(RunnableConfig, {"configurable": configurable})


def _prepare_run_config(
    *,
    thread_id: str | None,
    mcp_server_keys: list[str] | None,
    mode: str | None,
    model_id: str | None,
    session_id: str | None,
    enable_tracing: bool | None,
) -> RunnableConfig:
    base = _build_run_config(
        thread_id=thread_id,
        mcp_server_keys=mcp_server_keys,
    ) or {}
    run_config: dict[str, object] = dict(base)
    configurable = run_config.get("configurable")
    if isinstance(configurable, dict):
        run_config["configurable"] = dict(configurable)
    if enable_tracing is not True:
        return cast(RunnableConfig, run_config)
    configurable_payload = cast(dict[str, object], run_config.get("configurable") or {})
    if mode:
        configurable_payload["mode"] = mode
    if model_id:
        configurable_payload["model_id"] = model_id
    if configurable_payload:
        run_config["configurable"] = configurable_payload
    add_langfuse_callbacks(
        run_config,
        session_id=session_id or thread_id,
        user_id=thread_id,
    )
    return cast(RunnableConfig, run_config)


def _invoke_llm_with_optional_config(
    llm: object,
    messages: list[object],
    run_config: RunnableConfig | None,
) -> object:
    invoke = getattr(llm, "invoke")
    if run_config:
        try:
            return invoke(messages, config=run_config)
        except TypeError:
            return invoke(messages)
    return invoke(messages)


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return max(int(float(stripped)), 0)
        except ValueError:
            return 0
    return 0


def _normalize_usage(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage = cast(dict[str, object], raw)
    input_tokens = _to_int(
        usage.get("input") or usage.get("prompt_tokens") or usage.get("input_tokens")
    )
    output_tokens = _to_int(
        usage.get("output") or usage.get("completion_tokens") or usage.get("output_tokens")
    )
    total_tokens = _to_int(usage.get("total") or usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def _extract_usage(message: object) -> dict[str, int] | None:
    candidates: list[object] = []
    for attr in ("usage_metadata", "response_metadata", "additional_kwargs", "llm_output"):
        candidates.append(getattr(message, attr, None))
    if isinstance(message, dict):
        candidates.append(message)

    for candidate in candidates:
        parsed = _normalize_usage(candidate)
        if parsed:
            return parsed
        if isinstance(candidate, dict):
            mapping = cast(dict[str, object], candidate)
            for key in ("usage", "token_usage", "usage_metadata"):
                parsed = _normalize_usage(mapping.get(key))
                if parsed:
                    return parsed
    return None


def _pricing_for_model(model_id: str | None, input_tokens: int) -> _TokenPricing | None:
    model = (model_id or "").strip().lower()
    if not model:
        return None

    if "grok-code-fast-1" in model:
        return _TokenPricing(input_per_million=0.20, output_per_million=1.50)
    if "grok-4.2" in model:
        if input_tokens > 200_000:
            return _TokenPricing(input_per_million=4.00, output_per_million=12.00)
        return _TokenPricing(input_per_million=2.00, output_per_million=6.00)
    if "grok-4-fast" in model:
        if input_tokens > 128_000:
            return _TokenPricing(input_per_million=0.40, output_per_million=1.00)
        return _TokenPricing(input_per_million=0.20, output_per_million=0.50)
    if "grok-3-mini-fast" in model:
        return _TokenPricing(input_per_million=0.60, output_per_million=4.00)
    if "grok-3-fast" in model:
        return _TokenPricing(input_per_million=5.00, output_per_million=25.00)
    if "grok-3-mini" in model:
        return _TokenPricing(input_per_million=0.30, output_per_million=0.50)
    if "grok-3" in model or "grok-4" in model:
        return _TokenPricing(input_per_million=3.00, output_per_million=15.00)

    if "gemini-2.5-pro" in model:
        if input_tokens > 200_000:
            return _TokenPricing(input_per_million=2.50, output_per_million=15.00)
        return _TokenPricing(input_per_million=1.25, output_per_million=10.00)
    if "gemini-2.5-flash-lite" in model:
        return _TokenPricing(input_per_million=0.10, output_per_million=0.40)
    if "gemini-2.5-flash" in model:
        return _TokenPricing(input_per_million=0.30, output_per_million=2.50)

    if "gpt-oss-120b" in model:
        return _TokenPricing(input_per_million=0.15, output_per_million=0.60)
    if "gpt-oss-20b" in model:
        return _TokenPricing(input_per_million=0.07, output_per_million=0.30)
    return None


def _estimate_cost_usd(model_id: str | None, usage: dict[str, int]) -> tuple[float | None, str]:
    model = (model_id or "").strip().lower()
    if any(key in model for key in ("llama-4-scout", "llama-4-maverick", "large-meta")):
        return 0.0018 / 10_000.0, "transaction"
    if "llama-3.1-405b" in model:
        return 0.0267 / 10_000.0, "transaction"
    if any(key in model for key in ("llama-3.2-90b", "90b-vision")):
        return 0.005 / 10_000.0, "transaction"

    pricing = _pricing_for_model(model_id, usage.get("input", 0))
    if pricing is None:
        return None, "unknown"
    input_cost = (usage.get("input", 0) / 1_000_000.0) * pricing.input_per_million
    output_cost = (usage.get("output", 0) / 1_000_000.0) * pricing.output_per_million
    return input_cost + output_cost, "token"


def _emit_usage_observability(
    *,
    mode: str,
    model_id: str | None,
    session_id: str | None,
    thread_id: str | None,
    usage: dict[str, int] | None,
) -> tuple[dict[str, int] | None, float | None]:
    if usage is None:
        return None, None

    cost_usd, pricing_basis = _estimate_cost_usd(model_id, usage)
    logger.info(
        "llm_usage mode=%s model_id=%s session_id=%s thread_id=%s input_tokens=%d output_tokens=%d "
        "total_tokens=%d cost_usd=%.8f pricing_basis=%s",
        mode,
        model_id or "unknown",
        session_id or "unknown",
        thread_id or "unknown",
        usage.get("input", 0),
        usage.get("output", 0),
        usage.get("total", 0),
        cost_usd or 0.0,
        pricing_basis,
    )
    return usage, cost_usd


def _resolve_effective_mode(mode: str | None) -> str:
    explicit = str(mode or "").strip().lower()
    if explicit in {"direct", "rag", "mcp", "mixed"}:
        return explicit
    # Keep FastAPI/runtime behavior aligned with build_chat_config defaulting logic.
    from api.settings import get_settings

    settings = get_settings()
    enable_mcp_tools = bool(getattr(settings, "ENABLE_MCP_TOOLS", True))
    mcp_config = get_mcp_servers_config()
    if enable_mcp_tools and bool(mcp_config):
        return "mixed"
    return "rag"


class ChatRuntimeService:
    """Small service to execute direct, MCP, RAG, and mixed OCI chat modes."""

    def __init__(self, graph: Any = None) -> None:
        _ = graph
        self._thread_state: dict[str, dict[str, Any]] = {}

    async def run_chat(
        self,
        *,
        messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        collection_name: str | None,
        enable_reranker: bool | None,
        enable_tracing: bool | None,
        mode: str | None,
        mcp_server_keys: list[str] | None,
        stream: bool,
    ) -> dict[str, object]:
        _ = (
            session_id,
            collection_name,
            enable_reranker,
            enable_tracing,
            mcp_server_keys,
            stream,
        )
        normalized_mode = _resolve_effective_mode(mode)

        if normalized_mode == "mixed":
            latest_user_message = ""
            for item in messages:
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "")
                if role == "user":
                    latest_user_message = content.strip() or latest_user_message

            retrieval_tool = self._build_oracle_retrieval_tool(collection_name)
            resolved_model_id = model_id or get_llm().model_id
            run_cfg = _prepare_run_config(
                thread_id=thread_id,
                mcp_server_keys=mcp_server_keys,
                mode=normalized_mode,
                model_id=resolved_model_id,
                session_id=session_id,
                enable_tracing=enable_tracing,
            )
            tool_load_started = time.perf_counter()
            mcp_tools = await get_mcp_tools_async(
                server_keys=mcp_server_keys,
                run_config=run_cfg,
            )
            logger.info(
                "chat_runtime_mcp_tools_loaded mode=%s tool_count=%d elapsed_ms=%.1f",
                normalized_mode,
                len(mcp_tools),
                (time.perf_counter() - tool_load_started) * 1000,
            )
            final_answer, tools_used, tool_invocations = await get_mcp_answer_async(
                latest_user_message,
                model_id=resolved_model_id,
                tools=[retrieval_tool, *mcp_tools],
                run_config=run_cfg,
            )
            retrieval_state = getattr(retrieval_tool, "_retrieval_state", None)
            retrieval_docs = (
                cast(list[Document], retrieval_state.get("docs", []))
                if isinstance(retrieval_state, dict)
                else []
            )
            if not retrieval_docs and "oracle_retrieval" in tools_used and latest_user_message:
                retrieval_docs = self._retrieve_oracle_docs(
                    query=latest_user_message,
                    collection_name=collection_name,
                    k=8,
                )
            # Guardrail: if no MCP tools were used at all, fall back to direct
            # RAG retrieval+synthesis so doc-grounded questions don't bypass DB.
            # Do not override successful non-retrieval MCP tool answers (e.g. calculator).
            if latest_user_message and not tools_used:
                retrieval_docs = self._retrieve_oracle_docs(
                    query=latest_user_message,
                    collection_name=collection_name,
                    k=8,
                )
                if retrieval_docs:
                    rag_answer, rag_usage, resolved_model_id = await self._synthesize_rag_answer(
                        question=latest_user_message,
                        docs=retrieval_docs,
                        model_id=model_id,
                    )
                    _emit_usage_observability(
                        mode=normalized_mode,
                        model_id=resolved_model_id,
                        session_id=session_id,
                        thread_id=thread_id,
                        usage=rag_usage,
                    )
                    final_answer = rag_answer
            mixed_result: dict[str, object] = {
                "final_answer": final_answer,
                "error": None,
                "standalone_question": latest_user_message or None,
                "citations": self._citations_from_docs(retrieval_docs),
                "reranker_docs": self._serialize_docs(retrieval_docs),
                "context_usage": (
                    {"retrieved_docs_count": len(retrieval_docs)}
                    if retrieval_docs
                    else None
                ),
                "mcp_used": bool(tools_used),
                "mcp_tools_used": tools_used,
                "mcp_tool_invocations": tool_invocations,
            }
            if isinstance(model_id, str) and model_id.strip():
                mixed_result["model_id"] = model_id.strip()
            self._store_thread_state(thread_id, messages, mixed_result)
            return mixed_result
        if normalized_mode != "direct":
            if normalized_mode == "mcp":
                question = ""
                for item in messages:
                    role = str(item.get("role") or "").strip().lower()
                    content = str(item.get("content") or "")
                    if role == "user":
                        question = content.strip() or question

                resolved_model_id = model_id or get_llm().model_id
                run_cfg = _prepare_run_config(
                    thread_id=thread_id,
                    mcp_server_keys=mcp_server_keys,
                    mode=normalized_mode,
                    model_id=resolved_model_id,
                    session_id=session_id,
                    enable_tracing=enable_tracing,
                )
                tool_load_started = time.perf_counter()
                mcp_tools = await get_mcp_tools_async(
                    server_keys=mcp_server_keys,
                    run_config=run_cfg,
                )
                logger.info(
                    "chat_runtime_mcp_tools_loaded mode=%s tool_count=%d elapsed_ms=%.1f",
                    normalized_mode,
                    len(mcp_tools),
                    (time.perf_counter() - tool_load_started) * 1000,
                )
                answer, tools_used, tool_invocations = await get_mcp_answer_async(
                    question,
                    model_id=resolved_model_id,
                    tools=mcp_tools,
                    run_config=run_cfg,
                )
                mcp_result: dict[str, object] = {
                    "final_answer": answer,
                    "error": None,
                    "standalone_question": question or None,
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": None,
                    "mcp_used": bool(tools_used),
                    "mcp_tools_used": tools_used,
                    "mcp_tool_invocations": tool_invocations,
                }
                mcp_result["model_id"] = resolved_model_id
                self._store_thread_state(thread_id, messages, mcp_result)
                return mcp_result
            if normalized_mode == "rag":
                question = ""
                for item in messages:
                    role = str(item.get("role") or "").strip().lower()
                    content = str(item.get("content") or "")
                    if role == "user":
                        question = content.strip() or question

                docs = self._retrieve_oracle_docs(
                    query=question,
                    collection_name=collection_name,
                    k=5,
                )
                run_cfg = _prepare_run_config(
                    thread_id=thread_id,
                    mcp_server_keys=mcp_server_keys,
                    mode=normalized_mode,
                    model_id=model_id,
                    session_id=session_id,
                    enable_tracing=enable_tracing,
                )
                rag_answer, rag_usage, resolved_model_id = await self._synthesize_rag_answer(
                    question=question,
                    docs=docs,
                    model_id=model_id,
                    run_config=run_cfg,
                )
                emitted_usage, cost_usd = _emit_usage_observability(
                    mode=normalized_mode,
                    model_id=resolved_model_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    usage=rag_usage,
                )
                rag_result: dict[str, object] = {
                    "final_answer": rag_answer,
                    "error": None,
                    "standalone_question": question or None,
                    "citations": self._citations_from_docs(docs),
                    "reranker_docs": self._serialize_docs(docs),
                    "context_usage": None,
                    "mcp_used": False,
                    "mcp_tools_used": [],
                    "model_id": resolved_model_id,
                    "usage": emitted_usage,
                    "cost_usd": cost_usd,
                }
                self._store_thread_state(thread_id, messages, rag_result)
                return rag_result
            raise NotImplementedError(
                "run_chat currently only handles direct, mcp, rag, and mixed modes"
            )

        history: list[Any] = []
        latest_user_message = ""
        for item in messages:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "user":
                history.append(HumanMessage(content=content))
                latest_user_message = content.strip() or latest_user_message
            elif role == "assistant":
                history.append(AIMessage(content=content))

        run_cfg = _prepare_run_config(
            thread_id=thread_id,
            mcp_server_keys=mcp_server_keys,
            mode=normalized_mode,
            model_id=model_id,
            session_id=session_id,
            enable_tracing=enable_tracing,
        )
        llm = get_llm(model_id=model_id)
        response = await asyncio.to_thread(_invoke_llm_with_optional_config, llm, history, run_cfg)
        usage = _extract_usage(response)
        resolved_model_id = cast(str | None, getattr(llm, "model_id", None)) or model_id
        emitted_usage, cost_usd = _emit_usage_observability(
            mode=normalized_mode,
            model_id=resolved_model_id,
            session_id=session_id,
            thread_id=thread_id,
            usage=usage,
        )
        final_answer = str(getattr(response, "content", "") or "").strip()
        direct_result: dict[str, object] = {
            "final_answer": final_answer,
            "error": None,
            "standalone_question": latest_user_message or None,
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "model_id": resolved_model_id,
            "usage": emitted_usage,
            "cost_usd": cost_usd,
        }
        self._store_thread_state(thread_id, messages, direct_result)
        return direct_result

    async def stream_chat(
        self,
        *,
        messages: list[dict[str, object]],
        model_id: str | None,
        thread_id: str | None,
        session_id: str | None,
        collection_name: str | None,
        enable_reranker: bool | None,
        enable_tracing: bool | None,
        mode: str | None,
        mcp_server_keys: list[str] | None,
    ) -> AsyncIterator[dict[str, object]]:
        result = await self.run_chat(
            messages=messages,
            model_id=model_id,
            thread_id=thread_id,
            session_id=session_id,
            collection_name=collection_name,
            enable_reranker=enable_reranker,
            enable_tracing=enable_tracing,
            mode=mode,
            mcp_server_keys=mcp_server_keys,
            stream=True,
        )
        answer = str(result.get("final_answer") or "")
        if answer:
            yield {"type": "text", "delta": answer}

        references: dict[str, object] = {
            "standalone_question": result.get("standalone_question"),
            "citations": result.get("citations") or [],
            "reranker_docs": result.get("reranker_docs") or [],
        }
        if result.get("context_usage") is not None:
            references["context_usage"] = result["context_usage"]
        if result.get("mcp_used"):
            references["mcp_used"] = True
        if result.get("mcp_tools_used"):
            references["mcp_tools_used"] = result["mcp_tools_used"]
        invocations = result.get("mcp_tool_invocations")
        if isinstance(invocations, list) and invocations:
            references["mcp_tool_invocations"] = invocations
        if result.get("error"):
            references["error"] = result["error"]
        yield {"type": "references", "data": references}

    def _build_oracle_retrieval_tool(self, collection_name: str | None) -> StructuredTool:
        service = self

        class _OracleRetriever(BaseRetriever):
            collection: str
            retrieval_state: dict[str, object]

            def _get_relevant_documents(self, query: str, *, run_manager: object) -> list[Document]:
                _ = run_manager
                with get_pooled_connection() as conn:
                    embed_model = get_embedding_model()
                    vector_store = get_oracle_vs(conn, self.collection, embed_model)
                    docs = vector_store.similarity_search(query, 8)
                filtered = service._filter_retrieved_docs(query, docs)
                self.retrieval_state["docs"] = filtered
                return filtered

        state: dict[str, object] = {"docs": []}
        retriever = _OracleRetriever(
            collection=collection_name or "RAG_KNOWLEDGE_BASE",
            retrieval_state=state,
        )
        tool = create_retriever_tool(
            retriever,
            name="oracle_retrieval",
            description="Retrieve Oracle knowledge-base and documentation context for a user question.",
            response_format="content_and_artifact",
        )
        setattr(tool, "_retrieval_state", state)
        return tool

    def _retrieve_oracle_docs(
        self, *, query: str, collection_name: str | None, k: int
    ) -> list[Document]:
        collection = collection_name or "RAG_KNOWLEDGE_BASE"
        from api.settings import get_settings

        primary_mode = str(get_settings().RAG_SEARCH_MODE or "vector").strip().lower()
        fallback_modes: list[str] = []
        if primary_mode != "hybrid":
            fallback_modes.append("hybrid")
        if primary_mode != "text":
            fallback_modes.append("text")

        with get_pooled_connection() as conn:
            embed_model = get_embedding_model()
            docs = search_documents(
                conn=conn,
                collection_name=collection,
                embed_model=embed_model,
                query=query,
                top_k=k,
                search_mode=primary_mode,
            )
            if docs:
                logger.info(
                    "rag_retrieval mode=%s collection=%s docs=%d",
                    primary_mode,
                    collection,
                    len(docs),
                )
                return docs
            for mode in fallback_modes:
                docs = search_documents(
                    conn=conn,
                    collection_name=collection,
                    embed_model=embed_model,
                    query=query,
                    top_k=k,
                    search_mode=mode,
                )
                if docs:
                    logger.info(
                        "rag_retrieval_fallback mode=%s collection=%s docs=%d",
                        mode,
                        collection,
                        len(docs),
                    )
                    return docs

        logger.warning("rag_retrieval_no_docs collection=%s query_len=%d", collection, len(query or ""))
        return []

    async def _synthesize_rag_answer(
        self,
        *,
        question: str,
        docs: list[Document],
        model_id: str | None,
        run_config: RunnableConfig | None = None,
    ) -> tuple[str, dict[str, int] | None, str]:
        context = self._format_retrieved_docs(docs)
        answer_messages = [
            HumanMessage(content=RAG_ANSWER_PROMPT_TEMPLATE.format(question=question, context=context))
        ]
        llm = get_llm(model_id=model_id)
        final_message = await asyncio.to_thread(
            _invoke_llm_with_optional_config,
            llm,
            answer_messages,
            run_config,
        )
        resolved_model_id = cast(str | None, getattr(llm, "model_id", None)) or model_id or "unknown"
        return (
            str(getattr(final_message, "content", "") or "").strip(),
            _extract_usage(final_message),
            resolved_model_id,
        )

    def _filter_retrieved_docs(self, query: str, docs: list[Document]) -> list[Document]:
        query_terms = self._query_terms(query)
        if not docs or not query_terms:
            return docs

        required_overlap = 2 if len(query_terms) >= 3 else 1
        scored: list[tuple[int, Document]] = []
        for doc in docs:
            text_blob = " ".join(
                [
                    str(doc.page_content or ""),
                    str((doc.metadata or {}).get("source") or ""),
                    str((doc.metadata or {}).get("title") or ""),
                    str((doc.metadata or {}).get("file_name") or ""),
                ]
            ).lower()
            overlap = sum(1 for term in query_terms if term in text_blob)
            scored.append((overlap, doc))

        kept = [doc for overlap, doc in scored if overlap >= required_overlap]
        if kept:
            return kept[:5]

        best_overlap = max((overlap for overlap, _ in scored), default=0)
        if best_overlap > 0:
            best_docs = [doc for overlap, doc in scored if overlap == best_overlap]
            return best_docs[:3]

        return []

    def _query_terms(self, query: str) -> list[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "me",
            "of",
            "on",
            "or",
            "please",
            "tell",
            "that",
            "the",
            "to",
            "use",
            "what",
            "with",
        }
        terms = re.findall(r"[a-zA-Z0-9_]+", (query or "").lower())
        unique: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if len(term) < 3 or term in stopwords or term in seen:
                continue
            seen.add(term)
            unique.append(term)
        return unique

    def _latest_assistant_answer(
        self, thread_id: str | None, messages: list[dict[str, object]]
    ) -> str | None:
        if thread_id and thread_id in self._thread_state:
            prior_messages = list(self._thread_state[thread_id].get("messages") or [])
            for message in reversed(prior_messages):
                content = getattr(message, "content", None)
                if isinstance(message, AIMessage) and isinstance(content, str) and content.strip():
                    return content.strip()

        for item in reversed(messages[:-1]):
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "assistant" and content.strip():
                return content.strip()
        return None

    def _serialize_docs(self, docs: list[Document]) -> list[dict[str, object]]:
        return [
            {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
            for doc in docs
        ]

    def _citations_from_docs(self, docs: list[Document]) -> list[dict[str, object]]:
        return citations_from_documents(docs)

    def _format_retrieved_docs(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents were found."
        return "\n\n".join(f"[{idx}] {doc.page_content}" for idx, doc in enumerate(docs, start=1))

    async def get_state(self, run_config: dict[str, Any]) -> Any:
        thread_id = self._thread_id_from_run_config(run_config)
        values = self._thread_state.get(thread_id or "", {}) if thread_id else {}
        return type("StateSnapshot", (), {"values": values})()

    def get_state_values(self, run_config: dict[str, Any]) -> dict[str, Any] | None:
        thread_id = self._thread_id_from_run_config(run_config)
        if not thread_id:
            return None
        return self._thread_state.get(thread_id)

    async def delete_thread(self, thread_id: str) -> None:
        self._thread_state.pop(thread_id, None)

    def _thread_id_from_run_config(self, run_config: dict[str, Any]) -> str | None:
        configurable = run_config.get("configurable")
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
        thread_id = run_config.get("thread_id")
        if isinstance(thread_id, str) and thread_id.strip():
            return thread_id.strip()
        return None

    def _store_thread_state(
        self,
        thread_id: str | None,
        messages: list[dict[str, object]],
        result: dict[str, Any],
    ) -> None:
        if not thread_id:
            return

        prior_messages = list(self._thread_state.get(thread_id, {}).get("messages") or [])
        updated_messages = prior_messages + self._to_langchain_messages(messages)
        final_answer = str(result.get("final_answer") or "").strip()
        if final_answer:
            references: dict[str, object] = {}
            standalone = result.get("standalone_question")
            if isinstance(standalone, str) and standalone.strip():
                references["standalone_question"] = standalone.strip()
            citations = result.get("citations")
            if isinstance(citations, list):
                references["citations"] = citations
            reranker_docs = result.get("reranker_docs")
            if isinstance(reranker_docs, list):
                references["reranker_docs"] = reranker_docs
            context_usage = result.get("context_usage")
            if isinstance(context_usage, dict):
                references["context_usage"] = context_usage
            if result.get("mcp_used") is True:
                references["mcp_used"] = True
            mcp_tools_used = result.get("mcp_tools_used")
            if isinstance(mcp_tools_used, list):
                references["mcp_tools_used"] = [str(tool) for tool in mcp_tools_used if str(tool).strip()]
            mcp_inv = result.get("mcp_tool_invocations")
            if isinstance(mcp_inv, list) and mcp_inv:
                references["mcp_tool_invocations"] = mcp_inv
            error_value = result.get("error")
            if isinstance(error_value, str) and error_value.strip():
                references["error"] = error_value.strip()

            updated_messages.append(
                AIMessage(
                    content=final_answer,
                    additional_kwargs=references or {},
                    response_metadata=references or {},
                )
            )

        self._thread_state[thread_id] = {
            "messages": updated_messages,
            **result,
        }

    def _to_langchain_messages(self, messages: list[dict[str, object]]) -> list[Any]:
        converted: list[Any] = []
        for item in messages:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
        return converted

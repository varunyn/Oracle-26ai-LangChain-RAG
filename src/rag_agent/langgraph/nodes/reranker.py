"""Filtering and reranking of search results using OCI GenAI.

Uses the same OCI chat model as the rest of RAG (get_llm) with structured output
when available (RankedChunksResult), so no fragile JSON parsing. If structured
output fails, rerank is skipped and original retrieval order is preserved.

OCI also offers Cohere Rerank 3.5 via the RerankText API on dedicated AI clusters;
when available, a future backend could call that for dedicated rerank. This
implementation uses the on-demand chat model so it works without a dedicated cluster.
"""

import logging
import time
from collections.abc import Sequence
from typing import Protocol, cast

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from api.settings import get_settings

from ...agent_state import CitationEntry, DocSerializable, State
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.oci_models import get_llm
from ...metadata_schema import get_page_from_metadata, get_source_from_metadata
from ...prompts import RERANKER_TEMPLATE
from ...schemas import RankedChunksResult
from ...utils.context_window import estimate_tokens

logger = logging.getLogger(__name__)


class _StructuredLLM(Protocol):
    def invoke(
        self, messages: list[HumanMessage], *, config: RunnableConfig | None = None
    ) -> object: ...


def _truncate_for_rerank(text: str, *, max_tokens: int, model_id: str) -> str:
    if max_tokens <= 0:
        return text
    token_count = estimate_tokens(text, model_id)
    if token_count <= max_tokens:
        return text
    ratio = max_tokens / max(token_count, 1)
    target_len = max(64, int(len(text) * ratio))
    truncated = text[:target_len]
    while estimate_tokens(truncated, model_id) > max_tokens and len(truncated) > 64:
        truncated = truncated[: max(64, int(len(truncated) * 0.9))]
    return truncated


def _valid_rerank_indices(
    ranked_chunks: Sequence[object],
    *,
    num_docs: int,
    top_k: int,
) -> list[int]:
    """Return validated chunk indices in ranked order, capped to top_k entries.

    Each chunk may be a dict with 'index' or a Pydantic model with .index.
    Only indices in [0, num_docs) are kept; then the first top_k are returned.
    """
    if num_docs <= 0:
        return []
    seen: list[int] = []
    for chunk in ranked_chunks or []:
        if isinstance(chunk, dict):
            idx = chunk.get("index")
        else:
            idx = getattr(chunk, "index", None)
        if isinstance(idx, int) and 0 <= idx < num_docs and idx not in seen:
            seen.append(idx)
    return seen[:top_k]


class Reranker(Runnable[State, dict[str, object]]):
    """
    Implements a reranker using a LLM
    """

    def __init__(self):
        """
        Init
        """

    def generate_refs(self, docs: list[DocSerializable]) -> list[CitationEntry]:
        """
        Returns a list of reference dictionaries (source, page) for the given docs.
        Order is preserved: citations[i] corresponds to docs[i]; frontend uses this
        so that [1] in the answer maps to citations[0] and reranker_docs[0].
        """
        citations: list[CitationEntry] = []
        for doc in docs:
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            source = get_source_from_metadata(metadata)
            page = get_page_from_metadata(metadata)
            citations.append({"source": source, "page": page})
        return citations

    @staticmethod
    def get_reranked_docs(
        llm: object,
        query: str,
        retriever_docs: list[DocSerializable],
        model_id: str,
        max_doc_tokens: int,
        run_config: RunnableConfig | None = None,
    ) -> list[DocSerializable]:
        if max_doc_tokens <= 0:
            max_doc_tokens = 0
        chunks: list[str] = []
        for doc in retriever_docs:
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
            else:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if isinstance(content, str) and max_doc_tokens > 0:
                content = _truncate_for_rerank(
                    content,
                    max_tokens=max_doc_tokens,
                    model_id=model_id,
                )
            chunks.append(content if isinstance(content, str) else str(content))

        prompt_text = PromptTemplate(
            input_variables=["query", "chunks"],
            template=RERANKER_TEMPLATE,
        ).format(query=query, chunks=chunks)
        messages = [HumanMessage(content=prompt_text)]

        num_docs = len(retriever_docs)
        ranked_chunks_raw: Sequence[object] = []

        # Prefer OCI structured output (works with ChatOCIGenAI / Cohere provider)
        structured_output = getattr(llm, "with_structured_output", None)
        if callable(structured_output):
            try:
                structured_llm = cast(
                    _StructuredLLM,
                    structured_output(RankedChunksResult, method="json_schema"),
                )
                result = structured_llm.invoke(messages, config=run_config)
                if isinstance(result, RankedChunksResult) and result.ranked_chunks:
                    ranked_chunks_raw = result.ranked_chunks
                    logger.debug(
                        "Reranker: structured output (json_schema) num_chunks=%d",
                        len(ranked_chunks_raw),
                    )
            except Exception as e:
                logger.debug("Reranker structured output failed: %s", e)

        if not ranked_chunks_raw:
            logger.info("Reranker: structured output unavailable; skipping rerank")
            return retriever_docs

        indexes = _valid_rerank_indices(
            ranked_chunks_raw,
            num_docs=num_docs,
            top_k=get_settings().TOP_K,
        )
        return [retriever_docs[i] for i in indexes]

    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """
        Implements reranking logic.

        input: The agent state.
        """
        log_node_start("Rerank")
        t0 = time.perf_counter()
        # Rename parameter to avoid shadowing the config module
        run_config = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        enable_reranker = bool(configurable.get("enable_reranker", True))
        model_id = str(configurable.get("model_id") or get_settings().LLM_MODEL_ID)
        min_docs = int(get_settings().RERANKER_MIN_DOCS)
        max_doc_tokens = int(get_settings().RERANKER_MAX_DOC_TOKENS)

        user_request = str(input.get("standalone_question") or input.get("user_request") or "")
        retriever_docs = cast(list[DocSerializable], input.get("retriever_docs") or [])
        error = None

        if get_settings().DEBUG:
            logger.info("Reranker input state: %s", input)

        try:
            if retriever_docs:
                # there is something to rerank!
                if enable_reranker and len(retriever_docs) >= min_docs:
                    llm = get_llm(temperature=0.0)
                    reranked_docs = self.get_reranked_docs(
                        llm,
                        user_request,
                        retriever_docs,
                        model_id,
                        max_doc_tokens,
                        run_config=run_config,
                    )

                else:
                    reranked_docs = retriever_docs
            else:
                reranked_docs = []

        except Exception as e:
            logger.exception("Error in reranker: %s", e)
            error = str(e)
            # Fallback to original documents
            reranked_docs = retriever_docs

        # Get reference citations
        citations = self.generate_refs(reranked_docs)

        elapsed = time.perf_counter() - t0
        duration_ms = elapsed * 1000
        if enable_reranker and retriever_docs:
            logger.info(
                "Reranker completed in %.1fs (set ENABLE_RERANKER=False in config for faster RAG)",
                elapsed,
            )
        log_node_end("Rerank", duration_ms=duration_ms, doc_count=len(reranked_docs), error=error)
        out: dict[str, object] = {"reranker_docs": reranked_docs, "citations": citations}
        if error is not None:
            out["error"] = error
        return out

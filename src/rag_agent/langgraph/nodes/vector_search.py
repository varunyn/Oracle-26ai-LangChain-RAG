"""Semantic search in the agent using 23Ai Vector Search."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import cast

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from api.settings import get_settings

from ...agent_state import State
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.db_utils import get_pooled_connection
from ...infrastructure.oci_models import get_embedding_model
from ...infrastructure.retrieval import normalize_search_mode, search_documents
from ...utils.utils import docs_serializable

_SEARCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(getattr(get_settings(), "DB_SEARCH_MAX_WORKERS", 4))),
    thread_name_prefix="rag_search",
)

logger = logging.getLogger(__name__)


class SemanticSearch(Runnable[State, dict[str, object]]):
    """
    Implements Semantic Search for the Agent
    """

    def __init__(self):
        """
        Init
        """

    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        """
        This method invokes the vector search

        input: the agent state
        """
        log_node_start("Search")
        t0 = time.perf_counter()
        # Rename parameter to avoid shadowing the config module
        run_config = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        collection_name = str(
            configurable.get("collection_name") or get_settings().DEFAULT_COLLECTION
        )
        embed_model_type = str(
            configurable.get("embed_model_type") or get_settings().EMBED_MODEL_TYPE
        )
        search_mode = normalize_search_mode(
            str(configurable.get("search_mode") or get_settings().RAG_SEARCH_MODE)
        )

        relevant_docs = []
        error = None

        contextualized_question = str(
            input.get("standalone_question") or input.get("user_request") or ""
        )

        if get_settings().DEBUG:
            logger.info("Search question: %s", contextualized_question)

        try:
            embed_model = get_embedding_model(embed_model_type)
            # Explicit log: we are calling Oracle DB vector search (so 0 docs = empty collection or no matches, not "DB not called")
            logger.info(
                "Oracle retrieval: mode=%s, collection=%s, top_k=%d, query_preview=%s",
                search_mode,
                collection_name,
                get_settings().TOP_K,
                (
                    contextualized_question[:80] + "..."
                    if len(contextualized_question or "") > 80
                    else (contextualized_question or "")
                ),
            )

            timeout_sec = get_settings().DB_SEARCH_TIMEOUT_SEC

            def _do_search():
                with get_pooled_connection() as conn:
                    return search_documents(
                        conn=conn,
                        collection_name=collection_name,
                        embed_model=embed_model,
                        query=contextualized_question,
                        top_k=get_settings().TOP_K,
                        search_mode=search_mode,
                    )

            try:
                future = _SEARCH_EXECUTOR.submit(_do_search)
                relevant_docs = future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Vector search did not complete within {timeout_sec}s. Database may be stopped or unreachable; check DB_TCP_CONNECT_TIMEOUT and DB status."
                ) from None

            logger.debug("Retrieved %d documents from Oracle", len(relevant_docs))
            logger.info("Result from retrieval search: %d docs", len(relevant_docs))
            if len(relevant_docs) == 0:
                logger.info(
                    "Retrieval returned 0 docs (Oracle DB was queried). Check: collection '%s' is populated (e.g. run ingest) and embed model matches ingest.",
                    collection_name,
                )
            if get_settings().DEBUG:
                logger.debug("relevant_docs detail: %s", relevant_docs)

        except Exception as e:
            logger.exception("Error in vector_store.invoke: %s", e)
            error = str(e)
            # Record on current span so the error shows in Grafana Tempo / Oracle APM
            span = trace.get_current_span()
            if span.is_recording():
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

        duration_ms = (time.perf_counter() - t0) * 1000
        log_node_end("Search", duration_ms=duration_ms, doc_count=len(relevant_docs), error=error)
        # docs_serializable(relevant_docs)
        # convert the documents to a serializable format
        # to support the API
        return {
            "retriever_docs": docs_serializable(relevant_docs),
            "standalone_question": contextualized_question,
            "error": error,
        }

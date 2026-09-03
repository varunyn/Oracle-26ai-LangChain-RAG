"""FastMCP server exposing the typed Oracle knowledge retrieval contract."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, cast

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from pydantic import Field, ValidationError
from starlette.responses import JSONResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from api.settings import Settings, get_settings
from src.rag_agent.application.oracle_knowledge import (
    KnowledgeBaseListResult,
    ListDocumentsResult,
    OracleKnowledgeService,
    SearchKnowledgeRequest,
    SearchKnowledgeResult,
)
from src.rag_agent.infrastructure.oracle_knowledge import (
    KnowledgeReadinessProbe,
    build_oracle_knowledge_service,
)
from src.rag_agent.utils.logging_config import REQUEST_ID_CTX, setup_logging
from src.rag_agent.utils.otel_tracing import setup_otel_tracing_early

logger = logging.getLogger(__name__)
ORACLE_KNOWLEDGE_OTEL_SERVICE_NAME = "oracle-knowledge-mcp"


class RequestIdMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        request_id = context.fastmcp_context.request_id if context.fastmcp_context else None
        token = REQUEST_ID_CTX.set(str(request_id or uuid.uuid4()))
        try:
            return await call_next(context)
        finally:
            REQUEST_ID_CTX.reset(token)


def create_oracle_knowledge_server(
    service: OracleKnowledgeService, *, readiness_probe: KnowledgeReadinessProbe | None = None
) -> FastMCP:
    """Build a server around an injected service (also the contract-test seam)."""
    keys = ", ".join(service.knowledge_base_keys) or "none"
    instructions = (
        "Use search_knowledge for bounded evidence retrieval. "
        f"Allowed friendly knowledge-base keys: {keys}. "
        f"Limits: query length <= {service.max_query_length}, "
        f"results <= {service.max_result_limit}, "
        f"candidates <= {service.max_candidate_limit}, "
        f"metadata filters <= {service.max_metadata_filters}. "
        "The caller writes the final answer and owns citation presentation."
    )
    server = FastMCP(
        "Oracle Knowledge MCP",
        instructions=instructions,
    )
    server.add_middleware(RequestIdMiddleware())

    @server.custom_route("/health/live", methods=["GET"])
    async def health_live(_request: object) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @server.custom_route("/health/ready", methods=["GET"])
    async def health_ready(_request: object) -> JSONResponse:
        # Configuration and provider reachability are checked by service construction;
        # avoid chat-model calls and keep readiness failures secret-safe.
        ready, _reason = (
            await readiness_probe.check_async() if readiness_probe else (bool(service), "")
        )
        return JSONResponse(
            {
                "status": "ready" if ready else "not_ready",
                "error": None if ready else "service unavailable",
            },
            status_code=200 if ready else 503,
        )

    @server.tool
    async def search_knowledge(
        query: Annotated[
            str, Field(max_length=100000, description="Non-empty natural-language search query.")
        ],
        knowledge_base: Annotated[
            str | None,
            Field(max_length=128, description="Deployment-allowed friendly knowledge-base key."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100, description="Maximum evidence items.")] = 5,
        candidate_limit: Annotated[
            int | None,
            Field(ge=1, le=100, description="Maximum candidates inspected before result limiting."),
        ] = None,
        rerank: Annotated[
            bool | None, Field(description="Optional reranker override when deployment allows it.")
        ] = None,
        metadata_filters: Annotated[
            dict[str, str | int | bool] | None,
            Field(max_length=32, description="Supported filters: source, title, page."),
        ] = None,
    ) -> SearchKnowledgeResult:
        try:
            request = SearchKnowledgeRequest(
                query=query,
                knowledge_base=knowledge_base,
                limit=limit,
                candidate_limit=candidate_limit,
                rerank=rerank,
                metadata_filters=metadata_filters or {},
            )
        except ValidationError:
            return SearchKnowledgeResult(
                outcome="invalid_request",
                query=str(query).strip(),
                knowledge_base=(
                    knowledge_base if knowledge_base in service.knowledge_base_keys else None
                ),
                reranking_status="disabled",
                error="invalid search request",
            )
        return await service.search(request)

    @server.tool
    def list_knowledge_bases() -> KnowledgeBaseListResult:
        return service.list_knowledge_bases()

    @server.tool
    def list_documents(
        knowledge_base: Annotated[
            str | None,
            Field(max_length=128, description="Deployment-allowed friendly knowledge-base key."),
        ] = None,
    ) -> ListDocumentsResult:
        return service.list_documents(knowledge_base)

    return server


def build_service(settings: Settings | None = None) -> OracleKnowledgeService:
    settings = settings or get_settings()
    mapping = settings.ORACLE_KNOWLEDGE_BASES
    if settings.ORACLE_KNOWLEDGE_ALLOWED_KEYS is not None:
        allowed = set(settings.ORACLE_KNOWLEDGE_ALLOWED_KEYS)
        mapping = {key: collection for key, collection in mapping.items() if key in allowed}
    return build_oracle_knowledge_service(
        settings, knowledge_bases=mapping, default_key=settings.ORACLE_KNOWLEDGE_DEFAULT_KEY
    )


def main() -> None:
    settings = get_settings()
    os.environ.setdefault("OTEL_SERVICE_NAME", ORACLE_KNOWLEDGE_OTEL_SERVICE_NAME)
    os.environ["ENABLE_OTEL_TRACING"] = (
        "true" if settings.ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING else "false"
    )
    setup_logging()
    setup_otel_tracing_early(service_name=ORACLE_KNOWLEDGE_OTEL_SERVICE_NAME)
    server = create_oracle_knowledge_server(
        build_service(settings), readiness_probe=KnowledgeReadinessProbe(settings)
    )
    transport = settings.ORACLE_KNOWLEDGE_TRANSPORT
    kwargs: dict[str, Any] = {"transport": cast(Any, transport), "log_level": "INFO"}
    if transport != "stdio":
        kwargs.update(host=settings.ORACLE_KNOWLEDGE_HOST, port=settings.ORACLE_KNOWLEDGE_PORT)
    server.run(**kwargs)


if __name__ == "__main__":
    main()

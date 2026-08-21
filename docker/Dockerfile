# LangGraph Agent Server image. Build from project root:
# docker build -f docker/Dockerfile --target langgraph-runner .

# -----------------------------------------------------------------------------
# LangGraph builder: core deps + dev-only LangGraph CLI for local docker runtime
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS langgraph-builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --group dev --no-install-project

COPY langgraph.json ./langgraph.json
COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY mcp_servers ./mcp_servers

# -----------------------------------------------------------------------------
# Ingestion builder: adds heavy ingestion deps (docling, torch, OCR) on top
# -----------------------------------------------------------------------------
FROM langgraph-builder AS ingestion-builder

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --group dev --extra ingestion --no-install-project

# -----------------------------------------------------------------------------
# Base runner
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base-runner

WORKDIR /app

RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# LangGraph runner: minimal image without ingestion deps
# -----------------------------------------------------------------------------
FROM base-runner AS langgraph-runner

COPY --from=langgraph-builder /app/.venv /app/.venv
COPY --from=langgraph-builder --chown=appuser:appgroup /app/langgraph.json /app/langgraph.json
COPY --from=langgraph-builder --chown=appuser:appgroup /app/api /app/api
COPY --from=langgraph-builder --chown=appuser:appgroup /app/src /app/src
COPY --from=langgraph-builder --chown=appuser:appgroup /app/scripts /app/scripts
COPY --from=langgraph-builder --chown=appuser:appgroup /app/mcp_servers /app/mcp_servers

RUN mkdir /app/logs /app/uploaded_files /app/.langgraph_api \
    && chown -R appuser:appgroup /app/logs /app/uploaded_files /app/.langgraph_api

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 2024

USER appuser

CMD ["langgraph", "dev", "--config", "/app/langgraph.json", "--host", "0.0.0.0", "--port", "2024", "--allow-blocking", "--no-browser", "--no-reload"]

# -----------------------------------------------------------------------------
# Ingestion runner: adds system deps (OpenCV for OCR) and heavy ingestion deps
# -----------------------------------------------------------------------------
FROM base-runner AS ingestion-runner

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ingestion-builder /app/.venv /app/.venv
COPY --from=ingestion-builder --chown=appuser:appgroup /app/src /app/src
COPY --from=ingestion-builder --chown=appuser:appgroup /app/scripts /app/scripts

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

CMD ["python", "scripts/ingest_documents.py", "--help"]

FROM base-runner AS oracle-knowledge-runner
COPY --from=langgraph-builder /app/.venv /app/.venv
COPY --from=langgraph-builder --chown=appuser:appgroup /app/api /app/api
COPY --from=langgraph-builder --chown=appuser:appgroup /app/src /app/src
COPY --from=langgraph-builder --chown=appuser:appgroup /app/mcp_servers /app/mcp_servers
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9000
USER appuser
CMD ["python", "mcp_servers/oracle_knowledge.py"]

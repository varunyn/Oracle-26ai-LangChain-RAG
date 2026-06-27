# Backend (FastAPI RAG agent). Build from project root: docker build -f docker/Dockerfile .
# Multi-stage: builder installs deps with uv; runner is slim with app + venv only.

# -----------------------------------------------------------------------------
# Builder: install dependencies with uv (reproducible from lockfile)
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Optimize layer caching: deps first, then app code
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

# Install dependencies only (no project install; package = false in pyproject.toml)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy application code (api/, src/, scripts/, mcp_servers/)
COPY langgraph.json ./langgraph.json
COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY mcp_servers ./mcp_servers

# Full sync so env is complete (lockfile already satisfied)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# -----------------------------------------------------------------------------
# LangGraph builder: include dev-only LangGraph CLI for local docker runtime
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS langgraph-builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev --no-install-project

COPY langgraph.json ./langgraph.json
COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY mcp_servers ./mcp_servers

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev

FROM python:3.11-slim-bookworm AS base-runner

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# LangGraph runner: minimal image with LangGraph CLI available
# -----------------------------------------------------------------------------
FROM base-runner AS langgraph-runner

COPY --from=langgraph-builder --chown=appuser:appgroup /app/.venv /app/.venv
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
# Runner: minimal image, non-root user, run uvicorn
# -----------------------------------------------------------------------------
FROM base-runner AS runner

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/langgraph.json /app/langgraph.json
COPY --from=builder --chown=appuser:appgroup /app/api /app/api
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
COPY --from=builder --chown=appuser:appgroup /app/scripts /app/scripts
COPY --from=builder --chown=appuser:appgroup /app/mcp_servers /app/mcp_servers

RUN mkdir /app/logs /app/uploaded_files \
    && chown -R appuser:appgroup /app/logs /app/uploaded_files

ENV PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0

EXPOSE 3002

USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3002"]

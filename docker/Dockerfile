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
COPY --from=langchain_oci . /langchain-oracle/libs/oci
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy application code (api/, src/, scripts/, mcp_servers/)
COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY mcp_servers ./mcp_servers

# Full sync so env is complete (lockfile already satisfied)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-deps /langchain-oracle/libs/oci

# -----------------------------------------------------------------------------
# Runner: minimal image, non-root user, run uvicorn
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (match typical uid/gid for volume mounts)
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

# Copy virtualenv and application from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/api /app/api
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
COPY --from=builder --chown=appuser:appgroup /app/scripts /app/scripts
COPY --from=builder --chown=appuser:appgroup /app/mcp_servers /app/mcp_servers

RUN mkdir /app/logs /app/uploaded_files \
    && chown appuser:appgroup /app/logs /app/uploaded_files

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV HOST=0.0.0.0
EXPOSE 3002

USER appuser

# Production: no --reload
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3002"]

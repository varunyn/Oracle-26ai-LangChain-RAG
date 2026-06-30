# Observability — AGENTS.md (scoped to observability/)

## Overview

- Grafana + Tempo + Loki + OTEL collector and (optionally) Langfuse. Applies only to `observability/`.
- The main Compose observability profile exposes Grafana on `3051`, Tempo on `3200`, Loki on `3100`, and OTLP HTTP on `4318`.
- Langfuse has a separate Compose file under `observability/langfuse/` and its UI defaults to `3300`.

WHERE TO LOOK
- langfuse/: docker-compose and env example for Langfuse web/worker
- Other docker-compose.yml files per stack profile

COMMANDS (from repo root)
- Bring up stacks via orchestrated script:
  - uv run python scripts/manage_stacks.py up --stacks observability
  - uv run python scripts/manage_stacks.py down --stacks observability
  - uv run python scripts/manage_stacks.py status
- Langfuse: requires .env in observability/langfuse/ (see example). Services on http://localhost:3300

CONVENTIONS
- Use compose profiles managed by manage_stacks.py; do not run raw docker compose unless debugging
- Keep env files untracked; never commit secrets
- Logs/traces should include X-Request-ID from backend

ANTI-PATTERNS (FORBIDDEN)
- Modifying docker/compose/CI as part of unrelated feature work
- Committing .env files or credentials
- Diverging from manage_stacks.py profiles without documenting in root AGENTS.md

NOTES
- See root `AGENTS.md` and `docs/OBSERVABILITY.md` for routing traces/logs to Grafana/Tempo, OCI APM/Logging Analytics, and Langfuse.
- Ensure backend is started with OTEL enabled before validating traces

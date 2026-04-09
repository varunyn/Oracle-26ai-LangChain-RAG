#!/usr/bin/env bash
# Run the RAG API using this app's .venv (custom-rag-agent-app/.venv).
# Use this script so uv doesn't resolve a different project's venv (e.g. custom-rag-agent).
# If config has ENABLE_OBSERVABILITY_STACK=True, starts Loki/Tempo/OTel Collector/Grafana first (when Docker is available).
set -e
cd "$(dirname "$0")"

# Optionally start observability stack from config (set ENABLE_OBSERVABILITY_STACK=True to enable)
if ./.venv/bin/python -c "
import sys
try:
    import config
    if getattr(config, 'ENABLE_OBSERVABILITY_STACK', False):
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null; then
  if command -v docker >/dev/null 2>&1; then
    SERVICES=$(./.venv/bin/python - <<'PY'
import config

services = getattr(
    config,
    "OBSERVABILITY_STACK_SERVICES",
    ["loki", "tempo", "otel-collector", "grafana"],
)

if not isinstance(services, (list, tuple)) or not services:
    raise SystemExit("config.OBSERVABILITY_STACK_SERVICES must be a non-empty list[str]")

for s in services:
    if not isinstance(s, str) or not s.strip():
        raise SystemExit("config.OBSERVABILITY_STACK_SERVICES must be a non-empty list[str]")

print(" ".join(services))
PY
    )

    echo "Starting observability stack services: ${SERVICES}"
    docker compose --profile observability up -d ${SERVICES}
  else
    echo "Docker not found; skipping observability stack. API will run without Grafana/Loki/Tempo."
  fi
fi

# Limit graceful shutdown so reload doesn't wait forever for in-flight requests or OTel flush
exec ./.venv/bin/python -m uvicorn api.main:app --reload --port "${PORT:-3002}" --timeout-graceful-shutdown 15



# docker compose --profile observability stop loki tempo otel-collector grafana && docker compose --profile observability rm -f loki tempo otel-collector grafana 2>/dev/null || true

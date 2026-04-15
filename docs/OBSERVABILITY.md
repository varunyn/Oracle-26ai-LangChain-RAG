# Local Observability Stack

Run Loki, Tempo, the OTLP collector, and Grafana locally to inspect logs/traces without external services.

## 1. Prereqs

- Docker + docker compose
- `.env` copied from `.env.example` (see docs/CONFIGURATION.md)

## 2. Enable via config

```python
ENABLE_OBSERVABILITY_STACK = True  # run_api.sh starts the stack before the API
ENABLE_OTEL_TRACING = True         # send traces to the collector
```

Restart with `./run_api.sh`. When Docker isn’t available, set `ENABLE_OBSERVABILITY_STACK = False` and run the API only.

## 3. Manual control

```bash
# Stack only (preferred)
uv run python scripts/manage_stacks.py up --stacks observability

# Full app + stack (backend + frontend)
uv run python scripts/manage_stacks.py up --stacks core observability

# Stop & clean
uv run python scripts/manage_stacks.py down --stacks observability
```

Ports: Grafana 3051, Loki 3100, Tempo 3200, OTLP HTTP 4318.

## 4. Verify everything is up

```bash
uv run python scripts/manage_stacks.py status --stacks observability
curl -s http://localhost:3100/ready      # Loki
curl -s http://localhost:3200/ready      # Tempo
curl -s http://localhost:3051/api/health # Grafana
```

## 5. What flows through

- **Logs**: Application logs (INFO+) to `http://localhost:4318/v1/logs`, collector → Loki. Query with `{service_name="rag-api"}`.
- **Traces**: Spans to `http://localhost:4318/v1/traces`, collector → Tempo. Grafana Explore → Tempo → Service `rag-api`.
- **OCI Logging Analytics** is optional and filtered; see `LOGGING-ANALYTICS.md`.

## 6. Dashboards

- Grafana: <http://localhost:3051>
- Provisioned dashboards: **RAG API Overview** (HTTP health) and **RAG API – Pipeline & answers** (runtime flow). Use Explore for ad-hoc Loki/Tempo queries.

## 7. Troubleshooting

1. Run `bash observability/check-data.sh` to confirm collectors see data.
2. Ensure `ENABLE_OTEL_TRACING = True` and restart the API.
3. Set Grafana time range to “Last 15m” or “Last 1h”.
4. Check collector logs: `docker compose --profile observability logs otel-collector`.
5. If Loki exports time out, adjust `timeout` in `observability/otel-collector.yaml` and restart the collector.

## 8. Smoke tests without the API

```bash
uv run python scripts/manage_stacks.py up --stacks observability
uv run python observability/send-one-trace-and-log.py
bash observability/check-data.sh
```

## 9. Unit tests (no Docker)

```bash
uv run pytest tests/test_otel_logging.py tests/test_otel_tracing.py -v
```

They mock exporters, so the stack isn’t required.

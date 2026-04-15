# Trace a Request (local + OCI)

## 1. Enable tracing

- In `.env` set `ENABLE_OTEL_TRACING=true` (or export `ENABLE_OTEL_TRACING=1`).
- `OTEL_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) points to the OTLP consumer; default is the local collector `http://localhost:4318/v1/traces`.
- Tracing covers FastAPI requests, outgoing HTTP, LangChain runtime events, and LLM/tool spans. The app fails open—if the endpoint is down, the API still runs.

## 2. Local Grafana/Tempo

1. Start the observability stack (see `OBSERVABILITY.md` or the root `README.md` for the canonical local setup):

   ```bash
   uv run python scripts/manage_stacks.py up --stacks observability
   ```

2. In Grafana (`http://localhost:3051`) open **Explore → Tempo**.
3. Service name = `rag-api`, time range “Last 15 minutes”.

## 3. Oracle APM (OCI)

- Set `OTEL_TRACES_ENDPOINT = https://<data-upload-endpoint>/20200101/opentelemetry/private/v1/traces`.
- Add `OTEL_TRACES_HEADERS = {"Authorization": "dataKey <PRIVATE_KEY>"}` (or env var `OTEL_EXPORTER_OTLP_TRACES_HEADERS`).
- Use the Oracle quickstart [`apm_otel_langchain_oci.py`](https://github.com/oracle-quickstart/oci-observability-and-management/blob/master/examples/genai-inference-app-monitoring/apm_otel_langchain_oci.py) if you want GenAI-specific span attributes. Call `init(tracer)` before using OCI LangChain classes.

## 4. Troubleshooting

1. Confirm the API log shows `OpenTelemetry tracing initialized (service.name=rag-api)` after restart.
2. For debugging, check collector logs: `docker compose --profile observability logs otel-collector`.
3. In OCI APM, 400 “Data key missing/invalid” means the `dataKey` header is wrong.

## 5. Follow one request

1. Grab the `X-Request-ID` from the HTTP response.
2. In Grafana Loki: `Explore → Loki → {service_name="rag-api"} |= "<request_id>"` (time range Last 1h).
3. Look for key log lines:
   - `chat_in stream=... messages_count=...`
   - `chat_runtime_mcp_tools_loaded mode=... tool_count=...`
   - `stream_out references citations=... mcp_used=...`
   - `chat_out answer_len=... mcp_used=...`
4. Interpreting runtime logs:

| Field           | Notes                                   |
| --------------- | --------------------------------------- |
| `mode`          | `rag`, `mixed`, `mcp`, `direct`         |
| `tool_count`    | Number of MCP tools loaded for the turn |
| `mcp_used`      | Whether tools were actually invoked      |
| `citations`     | Number of normalized citations emitted  |

If mode/tool usage is unexpected (for example a RAG-only question calling MCP tools), inspect `chat_runtime_mcp_tools_loaded`, tool-call traces, and emitted `data-references`.

## 6. Langfuse SDK (optional)

Tracing is done **via the LangChain callback stack** (no API-level manual trace). When enabled, each `/api/langgraph/threads/{thread_id}/runs` request produces one Langfuse trace with nested spans for runtime steps, LLM calls, and tools.

1. **Bring up Langfuse (optional)**
   - Copy `observability/langfuse/.env.example` → `.env` and update all secrets
   - Start the stack: `docker compose -f observability/langfuse/docker-compose.yml up -d`
   - Langfuse UI is available at `http://localhost:3300` by default (MinIO console on `http://localhost:9091`)
2. Install the Python SDK (`pip install langfuse`) and set the Langfuse variables in `.env`:
   - `ENABLE_LANGFUSE_TRACING = True`
   - `LANGFUSE_HOST` (e.g., `http://localhost:3300` for the local stack)
   - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_TRACING_ENVIRONMENT` (optional, defaults to `development`)
3. Restart `./run_api.sh`. The chat route injects a Langfuse `CallbackHandler` into run config when Langfuse is enabled. Every invoke/stream sends a **single trace** with nested spans for LLM and tool execution, plus token usage where available.
4. Inspect the trace in Langfuse (Sessions → latest trace). The SDK runs fail-open—if Langfuse is offline, requests continue without blocking.
5. **Session vs thread**: The frontend sends a **session_id** (new per tab load/refresh, not persisted) and a **thread_id** (conversation continuity, persisted in localStorage). The backend passes `session_id` into the run config metadata (`langfuse_session_id`) so Langfuse groups traces into Sessions (one “browser visit”).

> Want to mix Grafana + OCI APM + Logging Analytics at the same time? See [`docs/OBSERVABILITY_ROUTING.md`](./OBSERVABILITY_ROUTING.md) for a full matrix of recipes and collector fan-out guidance.

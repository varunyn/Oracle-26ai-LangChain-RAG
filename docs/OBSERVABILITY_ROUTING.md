# Observability configuration guide (Local Grafana vs OCI APM vs OCI Logging Analytics)

This project supports **three distinct observability destinations**:

1. **Local Grafana/Loki/Tempo stack** (via the local OpenTelemetry Collector)
2. **OCI APM** (OTLP ingest for traces and optionally logs)
3. **OCI Logging Analytics** (separate log ingestion path)

The main confusion point: **`OTEL_TRACES_ENDPOINT` is a single endpoint**. If you want the same traces to go to _multiple_ backends, you typically do that **in the Collector**, not in the app.

---

## Mental model

### Two planes: “Stack runtime” vs “Export routing”

**A) Stack runtime (Docker)**

- `ENABLE_OBSERVABILITY_STACK=True` controls whether `run_api.sh` starts the local containers:
  - Loki, Tempo, OTEL Collector, Grafana
- This is about **starting services**, not about where the app exports.

**B) Export routing (App + Collector config)**

- `ENABLE_OTEL_TRACING`, `OTEL_TRACES_ENDPOINT`, `OTEL_LOGS_ENDPOINT`, etc. control **where telemetry is sent**.
- The Collector can **fan-out** (forward) telemetry to multiple downstream backends.

---

## Key configuration knobs (.env or environment)

### Traces (OTLP)

- `ENABLE_OTEL_TRACING: bool`
  - **False** → app does not export OTLP traces
  - **True** → app exports OTLP traces

- `OTEL_TRACES_ENDPOINT: str | None`
  - `None` → **default is local collector** `http://localhost:4318/v1/traces`
  - Non-None → send traces to that explicit OTLP endpoint (e.g., OCI APM ingest URL)

- `OTEL_TRACES_HEADERS: dict[str, str] | None`
  - Used for authenticated OTLP endpoints (OCI APM: `Authorization: dataKey <PRIVATE_KEY>`)

### Logs (OTLP)

- `OTEL_LOGS_ENDPOINT: str | None`
  - `None` → app uses its OTLP logs default (for this repo: local collector `http://localhost:4318/v1/logs`)
  - Non-None → send OTLP logs directly to the given endpoint

### Local stack

- `ENABLE_OBSERVABILITY_STACK: bool`
  - **True** → `run_api.sh` attempts `docker compose --profile observability up -d ...`
  - **False** → does not start docker stack

- `OBSERVABILITY_STACK_SERVICES: list[str]`
  - Controls which compose services `run_api.sh` starts when `ENABLE_OBSERVABILITY_STACK=True`.
  - Example: disable Grafana UI but keep Loki/Tempo/collector:
    - `["loki", "tempo", "otel-collector"]`

### OCI Logging Analytics (non-OTLP log shipper)

- `ENABLE_OCI_LOGGING_ANALYTICS: bool`
  - **True** → app additionally ships logs to OCI Logging Analytics (in parallel)
  - **False** → disabled

---

## What can run together?

### ✅ You can enable all three “systems” (recommended way)

**Goal**: Local Grafana/Tempo for dev + OCI APM for traces + OCI Logging Analytics for logs.

**Correct architecture**:

1. App → OTLP → **local Collector**
2. Collector → **Tempo** (local)
3. Collector → **OCI APM** (remote) ← fan-out here
4. App → **OCI Logging Analytics** (remote) ← separate pipeline

This requires **no app code changes**, but does require **Collector config** to add an OCI APM exporter.

### ❌ What you can’t do with a single OTEL endpoint

You cannot set `OTEL_TRACES_ENDPOINT` to two different OTLP endpoints at once.

If you set:

- `OTEL_TRACES_ENDPOINT=None` (local collector)

…then traces go to local collector _only_ unless the collector forwards them.

If you set:

- `OTEL_TRACES_ENDPOINT=<OCI APM URL>`

…then traces go directly to OCI APM and the local collector/Tempo won’t receive them (unless you also configure the app to export to the local collector separately, which would require extra code).

---

## Recipes (copy/paste)

### Recipe 1 — Local-only (Grafana/Tempo/Loki)

In `.env`:

```
ENABLE_OBSERVABILITY_STACK=true
ENABLE_OTEL_TRACING=true
# OTEL_TRACES_ENDPOINT and OTEL_LOGS_ENDPOINT unset → default local collector
ENABLE_OCI_LOGGING_ANALYTICS=false
```

Start:

```bash
./run_api.sh
```

Verify:

```bash
docker compose --profile observability ps
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:3051/api/health
```

### Recipe 2 — OCI APM only (no local stack)

In `.env`:

```
ENABLE_OBSERVABILITY_STACK=false
ENABLE_OTEL_TRACING=true
OTEL_TRACES_ENDPOINT=https://<oci-apm-otlp-endpoint>/.../v1/traces
OTEL_TRACES_HEADERS={"Authorization": "dataKey <PRIVATE_KEY>"}
ENABLE_OCI_LOGGING_ANALYTICS=false
```

Verify:

- check server logs for exporter errors
- check OCI APM for service.name=`rag-api`

### Recipe 3 — Local stack + OCI APM (fan-out via Collector)

In `.env`:

```
ENABLE_OBSERVABILITY_STACK=true
ENABLE_OTEL_TRACING=true
ENABLE_OCI_LOGGING_ANALYTICS=false
```

Then update Collector config (`observability/otel-collector.yaml`) to include an OCI APM exporter and wire it into the traces pipeline _in addition to_ Tempo.

**Example (current config + fan-out placeholder):**

```yaml
# observability/otel-collector.yaml (excerpt)
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true
  # Example additional exporter (replace with OCI APM details)
  # otlphttp/ociapm:
  #   endpoint: https://<oci-apm-otlp-endpoint>/.../v1/traces
  #   headers:
  #     Authorization: dataKey <PRIVATE_KEY>

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo, otlphttp/ociapm]
```

Use the OCI APM docs to set the correct endpoint and headers.

Verify:

```bash
docker compose --profile observability logs -f otel-collector
```

You should see successful export (no auth errors).

### Recipe 4 — Local stack + OCI Logging Analytics (dual-ship logs)

In `.env`:

```
ENABLE_OBSERVABILITY_STACK=true
ENABLE_OTEL_TRACING=true
ENABLE_OCI_LOGGING_ANALYTICS=true
LOGGING_ANALYTICS_NAMESPACE=...
LOGGING_ANALYTICS_LOG_GROUP_ID=ocid1.loggroup...
```

This yields:

- Traces/logs to local collector → Tempo/Loki
- Logs also shipped to Logging Analytics

---

## Common pitfalls

1. **“I enabled OCI APM but Tempo shows no traces”**
   - If `OTEL_TRACES_ENDPOINT` points directly to OCI APM, Tempo won’t see traces.
   - Fix: set endpoint back to local collector and forward from collector → APM.

2. **“`ENABLE_OBSERVABILITY_STACK=True` but nothing starts”**
   - Docker not running/installed.
   - Fix: start Docker Desktop; rerun `./run_api.sh`.

3. **“Collector is up but `/ready` returns 503”**
   - Loki/Tempo can take a moment to become ready after first boot.
   - Fix: retry after ~10–30 seconds.

4. **“OCI APM returns 400 data key missing/invalid”**
   - `OTEL_TRACES_HEADERS` must be `Authorization: dataKey <PRIVATE_KEY>`.

---

## Recommended default for local developers

For most contributors:

- Local observability **ON** (Grafana/Tempo/Loki) when debugging performance/traces
- Langfuse **OFF** unless doing LLM observability
- OCI APM / Logging Analytics **OFF** unless you explicitly need remote dashboards

---

## Note on server-owned chat memory

The `/api/chat` flow now persists `state["messages"]` via a checkpointer (SQLite by default). This does not change tracing/observability wiring, but you may see longer-lived conversation context reflected in LangChain/LangGraph spans and in Langfuse metadata (e.g., `mcp_used`, `mcp_tools_used`, `standalone_question`, and `context_usage`).

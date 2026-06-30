# OCI Logging Analytics Setup

Follow these steps to mirror OTLP logs from the API into Oracle Logging Analytics.

## 1. Configure `.env` (or environment)

```bash
ENABLE_OCI_LOGGING_ANALYTICS=true
LOGGING_ANALYTICS_NAMESPACE=<namespace>
LOGGING_ANALYTICS_LOG_GROUP_ID=<log_group_ocid>
LOGGING_ANALYTICS_LOG_SET=
LOGGING_ANALYTICS_RESOURCE_CATEGORY=rag-api
LOGGING_ANALYTICS_META_PROPERTIES=
LOGGING_ANALYTICS_MODE=auto  # or all
```

Restart the app (`make core-up`) after changing config.

## 2. Required OCI settings

OCI Console labels may say **Log Analytics** instead of **Logging Analytics**.

- **Namespace:** OCI Console → Observability & Management → Log Analytics → Administration → Service Details. Use the service namespace shown there.
- **Log Group OCID:** OCI Console → Observability & Management → Log Analytics → Administration → Log Groups → select your compartment → your log group → Actions → Copy OCID.
- **Log Set:** optional. Leave `LOGGING_ANALYTICS_LOG_SET` blank unless you intentionally route logs into a named log set.
- **Resource Category:** keep `LOGGING_ANALYTICS_RESOURCE_CATEGORY=rag-api` unless you want a different category in Log Explorer metadata.
- **Meta Properties:** optional semicolon-separated properties. Leave `LOGGING_ANALYTICS_META_PROPERTIES` blank to let the app send default `sourceName` and `resourceCategory` metadata.

### OCI CLI lookup

Use the tenancy/root compartment OCID to look up the namespace:

```bash
oci log-analytics namespace list --compartment-id "$TENANCY_OCID"
```

Then list log groups in the compartment where your Log Analytics log group lives:

```bash
oci log-analytics log-group list \
  --compartment-id "$COMPARTMENT_OCID" \
  --namespace-name "$LOGGING_ANALYTICS_NAMESPACE"
```

Use the returned log group OCID for `LOGGING_ANALYTICS_LOG_GROUP_ID`.

### IAM

The OCI principal configured by `OCI_CONFIG_FILE` and `OCI_PROFILE` must be allowed to upload logs to the target Log Analytics log group. A typical policy is:

```text
Allow group <group-name> to use loganalytics-log-group in compartment <compartment-name>
```

The `use loganalytics-log-group` permission includes the upload operation required by the verifier and runtime exporter.

## 3. Verify ingestion

1. Generate traffic (`curl http://localhost:3002/health` or a chat request).
2. Check API logs for `Log exporters: ... OCI Logging Analytics` at startup and ensure no `export failed` messages.
3. In OCI Console → Logging Analytics → Log Explorer, filter by the log group / OpenTelemetry Logs source and search for `service.name = "rag-api"`.

### CLI sanity check

From project root:

```bash
uv run python scripts/verify_oci_logging_analytics.py
```

`Upload OK` confirms IAM + region are correct.

## 4. Common issues

- **No logs:** Namespace or log group OCID mismatch, or IAM policy missing `LOG_ANALYTICS_LOG_GROUP_UPLOAD_LOGS` rights.
- **404 from OCI:** Wrong region in `local-config/oci/config`; set `OCI_PROFILE` / `OCI_CONFIG_FILE` to the project values.

## 5. Useful queries in Log Explorer

Start broad, then narrow once records appear:

```text
'Log Source' = 'OpenTelemetry Logs'
```

Count app summary events:

```text
'Log Source' = 'OpenTelemetry Logs'
| extract field = Attributes 'event_type, value=\{stringValue=(?P<event_type>[^\}]+)\}'
| where event_type != null
| stats count as records by event_type
| sort -records
```

Inspect chat response summaries:

```text
'Log Source' = 'OpenTelemetry Logs'
| extract field = Attributes 'event_type, value=\{stringValue=(?P<event_type>[^\}]+)\}'
| extract field = Attributes 'answer_len, value=\{(?:intValue|doubleValue)=(?P<answer_len_str>[0-9.]+)\}'
| extract field = Attributes 'mcp_tool_count, value=\{(?:intValue|doubleValue)=(?P<mcp_tool_count_str>[0-9.]+)\}'
| extract field = Attributes 'error, value=\{stringValue=(?P<error>[^\}]+)\}'
| where event_type = chat_out
| eval answer_len = toNumber(answer_len_str)
| eval mcp_tool_count = toNumber(mcp_tool_count_str)
| stats count as responses, avg(answer_len) as avg_answer_len, max(answer_len) as max_answer_len, avg(mcp_tool_count) as avg_mcp_tools by error
| sort -responses
```

The `chat_out` event intentionally sends lengths, error state, and MCP tool counts/names, not answer text or standalone question previews.

`langgraph_node` duration records are not emitted by the current runtime log path. Use Grafana/Tempo traces for node timing, or add explicit node-duration log events before building a Log Analytics dashboard around them.

Saved searches → dashboards can be created directly in the OCI console once these queries return data.

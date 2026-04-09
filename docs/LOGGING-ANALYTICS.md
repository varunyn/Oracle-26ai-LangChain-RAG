# OCI Logging Analytics Setup

Follow these steps to mirror OTLP logs from the API into Oracle Logging Analytics.

## 1. Configure `.env` (or environment)

```python
ENABLE_OCI_LOGGING_ANALYTICS = True
LOGGING_ANALYTICS_NAMESPACE = "<namespace>"
LOGGING_ANALYTICS_LOG_GROUP_ID = "<log_group_ocid>"
LOGGING_ANALYTICS_LOG_SET = None  # optional
LOGGING_ANALYTICS_RESOURCE_CATEGORY = "rag-api"
LOGGING_ANALYTICS_META_PROPERTIES = None
LOGGING_ANALYTICS_MODE = "auto"  # or "all"
```

Restart the API (`./run_api.sh`) after changing config.

## 2. Required OCI settings

- **Namespace:** Logging Analytics → Administration → Service Details.
- **Log Group OCID:** Logging Analytics → Log Groups → your group.
- Optional: log set, custom resource category, meta properties.

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

- `* | where Attributes.event_type = 'flow_trace' | stats count() by Attributes.answer_source`
- `* | where Attributes.event_type = 'chat_out' | stats avg(Attributes.answer_len) by Attributes.error`
- `* | where Attributes.event_type = 'langgraph_node' and Attributes.event = 'end' | stats count() by Attributes.node`

Saved searches → dashboards can be created directly in the OCI console once these queries return data.

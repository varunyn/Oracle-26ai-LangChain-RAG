#!/usr/bin/env python3
"""
Verify OCI Logging Analytics upload using the same config as the app.

Run from project root (so .env and local-config/oci/config are found):

  uv run python scripts/verify_oci_logging_analytics.py
  uv run python scripts/verify_oci_logging_analytics.py path/to/otlp.json

Success: prints "Upload OK". Failure: prints the exception (IAM, region, namespace, etc.).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

# Project root = parent of scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _main() -> int:
    sys.path.insert(0, str(_PROJECT_ROOT))

    from src.rag_agent.utils.logging_config import (
        LoggingAnalyticsExporter,
        _get_logging_analytics_settings,
        _normalize_otlp_json_for_oci,
    )

    settings = _get_logging_analytics_settings()
    if not settings:
        print(
            "OCI Logging Analytics is not enabled or config is incomplete. Check:\n"
            "  - ENABLE_OCI_LOGGING_ANALYTICS=true in .env (or env)\n"
            "  - LOGGING_ANALYTICS_NAMESPACE and LOGGING_ANALYTICS_LOG_GROUP_ID set\n"
            "  - No 'namespace/log group missing' in API startup log",
            file=sys.stderr,
        )
        return 1

    fixture_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _PROJECT_ROOT / "tests/fixtures/otlp_test_log.json"
    )
    if not fixture_path.is_file():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        return 1

    try:
        exporter = LoggingAnalyticsExporter(settings, "rag-api")
    except Exception as e:
        print(
            f"Failed to create OCI client (check OCI_CONFIG_FILE, OCI_PROFILE, key file): {e}",
            file=sys.stderr,
        )
        return 1

    with open(fixture_path) as f:
        payload_dict = json.load(f)
    _normalize_otlp_json_for_oci(payload_dict)
    json_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")

    kwargs = {
        "content_type": "application/json; charset=utf-8",
        "opc_request_id": str(uuid.uuid4()),
    }
    meta = exporter._meta_properties()
    if meta:
        kwargs["opc_meta_properties"] = meta
    if settings.log_set:
        kwargs["log_set"] = settings.log_set

    try:
        exporter._client.upload_otlp_logs(
            namespace_name=settings.namespace,
            opc_meta_loggrpid=settings.log_group_id,
            upload_otlp_logs_details=json_bytes,
            **kwargs,
        )
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return 1

    print(
        f"Upload OK. Check OCI Logging Analytics → Log Explorer (namespace={settings.namespace}, log group, last 5 min)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

import difflib
import json
from pathlib import Path
from typing import cast

import pytest

# Import the FastAPI app
from api.main import app

BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "openapi-baseline.json"
REFRESH_CMD = "uv run python scripts/export_openapi.py tests/fixtures/openapi-baseline.json"


def _canonical_json(data: object) -> str:
    """Return deterministic JSON string for comparison (sorted keys, 2-space indent)."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def test_openapi_matches_baseline():
    """
    Ensure current OpenAPI spec matches the committed baseline.

    On mismatch, show a readable unified diff and instructions to refresh the baseline
    explicitly (not automatic).
    """
    assert (
        BASELINE_PATH.is_file()
    ), f"Missing baseline at {BASELINE_PATH}. Generate it with:\n  {REFRESH_CMD}"

    # Load committed baseline
    baseline_obj = cast(dict[str, object], json.loads(BASELINE_PATH.read_text(encoding="utf-8")))

    # Generate current OpenAPI from the app, then canonicalize for deterministic diff
    current_obj = cast(dict[str, object], app.openapi())

    baseline_str = _canonical_json(baseline_obj)
    current_str = _canonical_json(current_obj)

    if baseline_str != current_str:
        diff = "".join(
            difflib.unified_diff(
                baseline_str.splitlines(keepends=True),
                current_str.splitlines(keepends=True),
                fromfile=str(BASELINE_PATH),
                tofile="<generated app.openapi()>",
            )
        )
        lines = [
            "OpenAPI spec has changed compared to the baseline.",
            "If this change is intentional, refresh the baseline with:",
            f"  {REFRESH_CMD}",
            "",
            "Diff (unified):",
            diff,
        ]
        pytest.fail("\n".join(lines))

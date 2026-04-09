import json
import subprocess
from pathlib import Path
from typing import TypedDict, cast

from api.main import app

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "docs" / "api" / "bruno" / "bruno-manifest.json"


class ManifestOperation(TypedDict):
    method: str
    path: str
    mode: str
    output: str


class BrunoManifest(TypedDict):
    operations: list[ManifestOperation]


def _openapi_operations() -> set[tuple[str, str]]:
    spec = app.openapi()
    ops: set[tuple[str, str]] = set()
    for path, methods in spec["paths"].items():
        for method in methods.keys():
            ops.add((method.upper(), path))
    return ops


def _manifest() -> BrunoManifest:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return cast(BrunoManifest, raw)


def test_api_docs_manifest_covers_all_openapi_operations() -> None:
    manifest = _manifest()
    manifest_ops = {(item["method"], item["path"]) for item in manifest["operations"]}
    assert manifest_ops == _openapi_operations()


def test_generated_bruno_entries_reference_real_openapi_operations() -> None:
    openapi_ops = _openapi_operations()
    manifest = _manifest()
    for item in manifest["operations"]:
        if item["mode"] != "generated":
            continue
        assert (item["method"], item["path"]) in openapi_ops
        assert (ROOT / "docs" / "api" / item["output"]).is_file()


def test_curated_operations_still_exist_in_openapi() -> None:
    openapi_ops = _openapi_operations()
    manifest = _manifest()
    curated: list[ManifestOperation] = [
        item for item in manifest["operations"] if item["mode"] == "curated"
    ]
    assert curated
    for item in curated:
        assert (item["method"], item["path"]) in openapi_ops
        assert (ROOT / "docs" / "api" / item["output"]).is_file()


def test_sync_api_docs_check_passes_when_artifacts_are_synced() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/sync_api_docs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

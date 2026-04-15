#!/usr/bin/env python3
"""Synchronize generated API docs artifacts and Bruno requests from FastAPI OpenAPI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

ROOT = PROJECT_ROOT
DOCS_API = ROOT / "docs" / "api"
GENERATED_DIR = DOCS_API / "generated"
BRUNO_ROOT = DOCS_API / "bruno" / "CustomRAGAgent"
MANIFEST_PATH = DOCS_API / "bruno" / "bruno-manifest.json"


def canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_openapi() -> dict[str, Any]:
    return dict(app.openapi())


def build_route_index(spec: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path, methods in sorted(spec.get("paths", {}).items()):
        for method, operation in sorted(methods.items()):
            operation_dict = dict(operation)
            entries.append(
                {
                    "path": path,
                    "method": method.upper(),
                    "operationId": operation_dict.get("operationId"),
                    "summary": operation_dict.get("summary"),
                    "tags": operation_dict.get("tags", []),
                }
            )
    return {"operations": entries}


def generate_endpoints_markdown(route_index: dict[str, Any]) -> str:
    lines = [
        "# Generated Endpoint Reference",
        "",
        "This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.",
        "Do not edit manually.",
        "",
    ]
    for entry in route_index["operations"]:
        tags = ", ".join(entry.get("tags") or []) or "untagged"
        summary = entry.get("summary") or ""
        lines.extend(
            [
                f"## {entry['method']} `{entry['path']}`",
                "",
                f"- operationId: `{entry.get('operationId')}`",
                f"- tags: {tags}",
                f"- summary: {summary}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_schemas_markdown(spec: dict[str, Any]) -> str:
    schemas = spec.get("components", {}).get("schemas", {})
    lines = [
        "# Generated Schema Reference",
        "",
        "This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.",
        "Do not edit manually.",
        "",
    ]
    for name in sorted(schemas):
        schema = schemas[name]
        schema_type = schema.get("type", "object")
        lines.append(f"## `{name}`")
        lines.append("")
        lines.append(f"- type: `{schema_type}`")
        required = schema.get("required") or []
        if required:
            lines.append(f"- required: {', '.join(f'`{r}`' for r in required)}")
        properties = schema.get("properties") or {}
        if properties:
            lines.append("")
            lines.append("### Properties")
            lines.append("")
            for prop_name in sorted(properties):
                prop = properties[prop_name]
                prop_type = prop.get("type") or "complex"
                description = prop.get("description") or ""
                lines.append(f"- `{prop_name}`: `{prop_type}` {description}".rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def default_body_for_operation(path: str) -> dict[str, Any] | None:
    if path == "/api/feedback":
        return {
            "question": "What is Oracle vector search?",
            "answer": "It is a database-native vector retrieval capability.",
            "feedback": 5,
        }
    if path == "/api/suggestions":
        return {
            "last_message": "Oracle vector search combines embeddings with document retrieval.",
            "model": None,
        }
    return None


def bruno_request_text(entry: dict[str, Any]) -> str:
    method = entry["method"].lower()
    name = entry["name"]
    path = entry["path"]
    body_kind = entry.get("bodyKind", "none")
    lines = [
        "meta {",
        f"  name: {name}",
        "  type: http",
        f"  seq: {entry.get('seq', 1)}",
        "}",
        "",
        f"{method} {{",
        f"  url: {{baseUrl}}{path}",
        f"  body: {body_kind}",
        "}",
    ]
    if body_kind == "json":
        lines.extend(
            [
                "",
                "headers {",
                "  Content-Type: application/json",
                "}",
                "",
                "body:json {",
                json.dumps(entry["body"], indent=2),
                "}",
            ]
        )
    elif body_kind == "multipart-form":
        lines.extend(
            [
                "",
                "body:multipart-form {",
                "  collection_name: {{defaultCollection}}",
                "  files: @./example.pdf",
                "}",
            ]
        )
    return "\n".join(lines) + "\n"


def generate_bruno_files(route_index: dict[str, Any], manifest: dict[str, Any]) -> dict[Path, str]:
    operations = {(item["method"], item["path"]): item for item in route_index["operations"]}
    outputs: dict[Path, str] = {}

    outputs[BRUNO_ROOT / "bruno.json"] = canonical_json(
        {"version": "1", "name": "CustomRAGAgent", "type": "collection"}
    )
    outputs[BRUNO_ROOT / "environments.bru"] = (
        "vars {\n"
        "  baseUrl: http://127.0.0.1:3002\n"
        "  defaultCollection: RAG_KNOWLEDGE_BASE\n"
        "  defaultModel: cohere.command-r-plus\n"
        "  threadId: bruno-local-thread\n"
        "  sessionId: bruno-local-session\n"
        "}\n"
    )

    for item in manifest["operations"]:
        key = (item["method"], item["path"])
        if key not in operations:
            raise ValueError(f"Manifest operation missing from OpenAPI: {key}")
        if item["mode"] != "generated":
            continue
        out_path = DOCS_API / item["output"]
        body_kind = item.get("bodyKind", "none")
        body = None
        if body_kind == "json":
            body = default_body_for_operation(item["path"])
            if body is None:
                raise ValueError(f"No default JSON body available for {item['path']}")
        outputs[out_path] = bruno_request_text(
            {
                "method": item["method"],
                "name": item["name"],
                "path": item["path"],
                "seq": item.get("seq", 1),
                "bodyKind": body_kind,
                "body": body,
            }
        )
    return outputs


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_or_check(outputs: dict[Path, str], check: bool) -> list[str]:
    mismatches: list[str] = []
    for path, expected in outputs.items():
        if path.exists():
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            mismatches.append(str(path.relative_to(ROOT)))
        if not check and (not path.exists() or path.read_text(encoding="utf-8") != expected):
            ensure_parent(path)
            path.write_text(expected, encoding="utf-8")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Check for drift without writing files"
    )
    args = parser.parse_args()

    spec = load_openapi()
    route_index = build_route_index(spec)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[Path, str] = {
        GENERATED_DIR / "openapi.json": canonical_json(spec),
        GENERATED_DIR / "route-index.json": canonical_json(route_index),
        GENERATED_DIR / "endpoints.md": generate_endpoints_markdown(route_index),
        GENERATED_DIR / "schemas.md": generate_schemas_markdown(spec),
    }
    outputs.update(generate_bruno_files(route_index, manifest))

    mismatches = write_or_check(outputs, args.check)
    if mismatches:
        if args.check:
            print("API docs sync drift detected in:")
            for item in mismatches:
                print(f"- {item}")
            print("Run: uv run python scripts/sync_api_docs.py")
            return 1
        print(f"Updated {len(mismatches)} file(s).")
    else:
        print("API docs artifacts are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

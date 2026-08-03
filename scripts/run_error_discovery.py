#!/usr/bin/env python3
"""Build and serve the local error-discovery review app.

Usage: uv run python scripts/run_error_discovery.py [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / ".local-data" / "langgraph-checkpoints.sqlite"
DATA = ROOT / "error_discovery_data"
STATIC = ROOT / "scripts" / "error_discovery_app.html"


def plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "content"):
        return plain(getattr(value, "content"))
    return str(value)


def build_records() -> list[dict[str, Any]]:
    serializer = JsonPlusSerializer()
    con = sqlite3.connect(DB)
    latest: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for thread_id, checkpoint_id, typ, blob, metadata in con.execute(
        "select thread_id, checkpoint_id, type, checkpoint, metadata from checkpoints"
    ):
        try:
            checkpoint = serializer.loads_typed((typ, blob))
            meta = json.loads(metadata)
        except Exception:
            continue
        if checkpoint.get("ts", "") >= latest.get(thread_id, ({"ts": ""}, {}))[0].get("ts", ""):
            latest[thread_id] = (checkpoint, meta)
    records: list[dict[str, Any]] = []
    for thread_id, (checkpoint, meta) in latest.items():
        state = checkpoint.get("channel_values", {})
        messages = state.get("messages", [])
        turns = []
        for index, message in enumerate(messages):
            role = message.__class__.__name__.replace("Message", "")
            content = plain(getattr(message, "content", ""))
            if isinstance(content, list):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            turns.append({"id": f"{thread_id}-{index}", "role": role, "content": str(content)})
        user_request = str(state.get("user_request") or next((m["content"] for m in turns if m["role"] == "Human"), ""))
        final_answer = state.get("final_answer")
        if final_answer is None:
            final_answer = next((m["content"] for m in reversed(turns) if m["role"] == "AI"), "")
        answer = plain(final_answer)
        if not isinstance(answer, str):
            answer = json.dumps(answer, indent=2, ensure_ascii=False)
        citations = plain(state.get("citations") or [])
        mode = meta.get("mode") or state.get("mode") or "unknown"
        chars = sum(len(t["content"]) for t in turns)
        records.append({
            "id": thread_id,
            "title": user_request[:72] or "Untitled trace",
            "topic": mode,
            "label": "error-candidate" if not answer.strip() or answer.strip() == user_request.strip() else "unreviewed",
            "mode": mode,
            "timestamp": checkpoint.get("ts", ""),
            "turns": turns,
            "user_request": user_request,
            "final_answer": answer,
            "citations": citations,
            "retriever_docs": plain(state.get("retriever_docs") or []),
            "reranker_docs": plain(state.get("reranker_docs") or []),
            "error": plain(state.get("error")),
            "metadata": {k: plain(v) for k, v in meta.items() if k not in {"langfuse_public_key", "langfuse_secret_key"}},
            "features": [len(turns), chars, len(citations), len(state.get("retriever_docs") or []), len(state.get("reranker_docs") or []), {"direct": 0, "mixed": 1, "rag": 2, "mcp": 3, "unknown": 4}.get(mode, 4)],
        })
    return sorted(records, key=lambda r: r["timestamp"])


def percentile_flags(records: list[dict[str, Any]]) -> None:
    lengths = [len(r["turns"]) for r in records]
    chars = [sum(len(t["content"]) for t in r["turns"]) for r in records]
    for r in records:
        n = len(r["turns"])
        c = sum(len(t["content"]) for t in r["turns"])
        flags = []
        if n >= max(lengths) and n > 1:
            flags.append(f"{n} turns")
        if c >= max(chars) and c > 300:
            flags.append("long trace")
        if not r["final_answer"].strip():
            flags.append("missing final answer")
        if r["error"]:
            flags.append("runtime error")
        r["flags"] = flags


def select_samples(records: list[dict[str, Any]], count: int = 22) -> list[dict[str, Any]]:
    random.seed(20260723)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["mode"], min(len(r["turns"]), 5)), []).append(r)
    chosen: list[dict[str, Any]] = []
    for group in groups.values():
        chosen.append(min(group, key=lambda r: (len(r["final_answer"]), r["id"])))
    remaining = [r for r in records if r not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen[:count]


def write_json(name: str, value: Any) -> None:
    (DATA / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare() -> None:
    DATA.mkdir(exist_ok=True)
    records = build_records()
    percentile_flags(records)
    samples = select_samples(records)
    # Small deterministic projection: feature axes preserve mode/length/answer coverage.
    graph = []
    for i, r in enumerate(records):
        f = r["features"]
        graph.append({"id": r["id"], "x": (f[1] / 900 + f[2] * 0.7) % 10, "y": (f[0] * 1.5 + f[3] * 0.4 + f[5] * 1.2) % 10, "cluster": (f[5] + f[0]) % 7, "sample": r in samples})
    write_json("records.json", records)
    write_json("samples.json", samples)
    write_json("graph.json", graph)
    write_json("annotations.json", []) if not (DATA / "annotations.json").exists() else None
    write_json("patterns.json", {}) if not (DATA / "patterns.json").exists() else None
    write_json("suggestions.json", []) if not (DATA / "suggestions.json").exists() else None
    print(f"Prepared {len(records)} records and {len(samples)} initial samples in {DATA}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)

    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        files = {"/api/records": "records.json", "/api/samples": "samples.json", "/api/graph": "graph.json", "/api/annotations": "annotations.json", "/api/patterns": "patterns.json", "/api/suggestions": "suggestions.json"}
        if path in files:
            self._json(json.loads((DATA / files[path]).read_text(encoding="utf-8")))
        elif path == "/":
            body = STATIC.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        mapping = {"/api/samples": "samples.json", "/api/annotations": "annotations.json", "/api/patterns": "patterns.json", "/api/suggestions": "suggestions.json"}
        if path not in mapping:
            self._json({"error": "not found"}, 404); return
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or "null")
        write_json(mapping[path], value)
        self._json({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8765); args = parser.parse_args()
    prepare()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Error discovery app: http://127.0.0.1:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()

"""Restore local Agent Server thread registry entries from SQLite checkpoints.

Usage:
    uv run python scripts/run_restore_langgraph_thread_registry.py

The local LangGraph dev runtime stores graph checkpoints in
`LANGGRAPH_SQLITE_PATH` and stores the thread/run registry in `.langgraph_api`.
If `.langgraph_api` was not mounted in Docker, existing checkpoints can remain
on disk while `/threads/<id>/state` returns 404. This script recreates missing
thread registry entries so the Agent Server can read the existing checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from uuid import UUID

from langgraph_sdk import get_sync_client


def _default_sqlite_path() -> Path:
    docker_host_path = Path("local-data/langgraph-checkpoints.sqlite")
    if docker_host_path.exists():
        return docker_host_path
    return Path(
        os.environ.get(
            "LANGGRAPH_SQLITE_PATH",
            ".local-data/langgraph-checkpoints.sqlite",
        )
    )


def _checkpoint_thread_ids(sqlite_path: Path) -> list[str]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite checkpoint file not found: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            """
            SELECT thread_id, MAX(rowid) AS latest_rowid
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY latest_rowid DESC
            """
        ).fetchall()
    return [str(thread_id) for thread_id, _ in rows if str(thread_id).strip()]


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _existing_thread_ids(api_url: str, *, limit: int) -> set[str]:
    client = get_sync_client(url=api_url)
    existing: set[str] = set()
    offset = 0
    while True:
        threads = client.threads.search(
            limit=limit,
            offset=offset,
            select=["thread_id"],
        )
        if not threads:
            return existing
        existing.update(str(thread["thread_id"]) for thread in threads)
        if len(threads) < limit:
            return existing
        offset += limit


def restore_thread_registry(
    *,
    api_url: str,
    sqlite_path: Path,
    graph_id: str,
    page_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    checkpoint_thread_ids = _checkpoint_thread_ids(sqlite_path)
    skipped_non_uuid = [thread_id for thread_id in checkpoint_thread_ids if not _is_uuid(thread_id)]
    for thread_id in skipped_non_uuid:
        print(f"skipped non-uuid {thread_id}")
    checkpoint_thread_ids = [
        thread_id for thread_id in checkpoint_thread_ids if _is_uuid(thread_id)
    ]
    existing_thread_ids = _existing_thread_ids(api_url, limit=page_size)
    missing_thread_ids = [
        thread_id for thread_id in checkpoint_thread_ids if thread_id not in existing_thread_ids
    ]

    if dry_run:
        for thread_id in missing_thread_ids:
            print(f"missing {thread_id}")
        return len(checkpoint_thread_ids), len(missing_thread_ids)

    client = get_sync_client(url=api_url)
    for thread_id in missing_thread_ids:
        client.threads.create(
            thread_id=thread_id,
            graph_id=graph_id,
            if_exists="do_nothing",
        )
        print(f"restored {thread_id}")
    return len(checkpoint_thread_ids), len(missing_thread_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024"),
        help="LangGraph Agent Server URL.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(_default_sqlite_path()),
        help="Path to the SQLite checkpoint file.",
    )
    parser.add_argument(
        "--graph-id",
        default="chat_agent",
        help="Graph id to attach to restored thread registry entries.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total, restored = restore_thread_registry(
        api_url=args.api_url,
        sqlite_path=Path(args.sqlite_path),
        graph_id=args.graph_id,
        page_size=args.page_size,
        dry_run=args.dry_run,
    )
    action = "would restore" if args.dry_run else "restored"
    print(f"{action} {restored} missing thread registry entries from {total} checkpoint threads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

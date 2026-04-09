#!/usr/bin/env python3
"""
Truncate RAG table(s) (e.g. RAG_KNOWLEDGE_BASE).

Uses config.CONNECT_ARGS and config.COLLECTION_LIST. Requires --yes to confirm.
Run from project root: uv run scripts/truncate_table.py [--table NAME] [--yes]
"""

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import oracledb

from api.settings import get_settings


def truncate_table(conn: oracledb.Connection, table_name: str) -> int:
    """
    Truncate a single table. Returns row count before truncate (if available).

    Args:
        conn: Oracle DB connection.
        table_name: Name of the table to truncate.

    Returns:
        Number of rows before truncate (0 if not available).
    """
    cursor = conn.cursor()
    try:
        # Optional: get count before truncate (for reporting)
        name_upper = table_name.upper()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {name_upper}")
            (count,) = cursor.fetchone()
        except oracledb.Error:
            count = 0
        cursor.execute(f"TRUNCATE TABLE {name_upper}")
        conn.commit()
        return int(count) if count is not None else 0
    finally:
        cursor.close()


def main() -> int:
    """Run truncate: parse args, connect, truncate, report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Truncate RAG table(s) (e.g. RAG_KNOWLEDGE_BASE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/truncate_table.py --yes
  uv run scripts/truncate_table.py --table RAG_KNOWLEDGE_BASE --yes
  uv run scripts/truncate_table.py --all --yes
        """,
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table to truncate (default: config.DEFAULT_COLLECTION)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Truncate all tables in config.COLLECTION_LIST",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm truncation (required to run)",
    )
    args = parser.parse_args()

    if not args.yes:
        print("Error: --yes is required to confirm truncation.")
        return 1

    if args.all:
        tables = getattr(get_settings(), "COLLECTION_LIST", None) or [
            getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")
        ]
    else:
        tables = [args.table or getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")]

    connect_args = cast(Mapping[str, str | None], get_settings().CONNECT_ARGS)
    conn = oracledb.connect(**connect_args)
    try:
        for table_name in tables:
            table_name = table_name.strip()
            if not table_name:
                continue
            try:
                count = truncate_table(conn, table_name)
                print(f"Truncated {table_name} (had {count} row(s)).")
            except oracledb.Error as e:
                print(f"Error truncating {table_name}: {e}", file=sys.stderr)
                return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

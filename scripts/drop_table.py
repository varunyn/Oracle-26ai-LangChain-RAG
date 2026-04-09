#!/usr/bin/env python3
"""
Drop RAG vector store table(s) (e.g. RAG_KNOWLEDGE_BASE_TEST).

Uses config.CONNECT_ARGS. Requires --yes to confirm.
Run from project root: uv run scripts/drop_table.py [--table NAME] [--yes]
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


def drop_table(conn: oracledb.Connection, table_name: str) -> bool:
    """
    Drop a single table if it exists.

    Args:
        conn: Oracle DB connection.
        table_name: Name of the table to drop.

    Returns:
        True if table was dropped, False if it did not exist.
    """
    cursor = conn.cursor()
    try:
        table_name_upper = table_name.upper()
        cursor.execute(
            "SELECT COUNT(*) FROM user_tables WHERE UPPER(table_name) = UPPER(:1)",
            [table_name_upper],
        )
        (exists,) = cursor.fetchone()
        if not exists:
            print(f"Table {table_name_upper} does not exist; nothing to drop.")
            return False
        cursor.execute(f"DROP TABLE {table_name_upper}")
        conn.commit()
        print(f"Dropped table {table_name_upper}.")
        return True
    finally:
        cursor.close()


def main() -> int:
    """Run drop: parse args, connect, drop, report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Drop RAG vector store table(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/drop_table.py --table RAG_KNOWLEDGE_BASE_TEST --yes
  uv run scripts/drop_table.py --table RAG_KNOWLEDGE_BASE --yes
        """,
    )
    parser.add_argument(
        "--table",
        required=True,
        help="Table name to drop (e.g. RAG_KNOWLEDGE_BASE_TEST)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm drop (required to run)",
    )
    args = parser.parse_args()

    if not args.yes:
        print("Error: --yes is required to confirm drop.")
        return 1

    table_name = args.table.strip()
    if not table_name:
        print("Error: --table is required.", file=sys.stderr)
        return 1

    connect_args = cast(Mapping[str, str | None], get_settings().CONNECT_ARGS)
    conn = oracledb.connect(**connect_args)
    try:
        drop_table(conn, table_name)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

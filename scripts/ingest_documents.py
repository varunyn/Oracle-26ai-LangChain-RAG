#!/usr/bin/env python3
"""CLI wrapper for document ingestion into OracleVS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_agent.ingestion import (
    DEFAULT_TABLE_NAME,
    SUPPORTED_EXTENSIONS,
    copy_file_to_uploaded,
    ensure_uploaded_files_dir,
    get_project_root,
    load_document_with_langchain,
    load_documents_from_dir,
    load_documents_from_files,
    populate_from_dir,
    populate_from_files,
    process_file_paths,
)

__all__ = [
    "DEFAULT_TABLE_NAME",
    "SUPPORTED_EXTENSIONS",
    "copy_file_to_uploaded",
    "ensure_uploaded_files_dir",
    "get_project_root",
    "load_document_with_langchain",
    "load_documents_from_dir",
    "load_documents_from_files",
    "populate_from_dir",
    "populate_from_files",
    "process_file_paths",
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the configured Oracle vector collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/ingest_documents.py --files document.pdf readme.md
  python scripts/ingest_documents.py --dir ./documents

Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs="+", help="File path(s) to process (PDF, HTML, TXT, MD)")
    group.add_argument("--dir", help="Directory path to load all supported files from")
    parser.add_argument(
        "--table",
        default=None,
        help=f"Table name (default: config.DEFAULT_COLLECTION = {DEFAULT_TABLE_NAME})",
    )
    args = parser.parse_args()
    table_name = args.table or DEFAULT_TABLE_NAME

    if args.files:
        populate_from_files(args.files, table_name=table_name)
    else:
        populate_from_dir(args.dir, table_name=table_name)


if __name__ == "__main__":
    main()

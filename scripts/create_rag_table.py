#!/usr/bin/env python3
"""
Create an OracleVS-compatible RAG table (default: RAG_KNOWLEDGE_BASE).

This script creates the table shape used by this repo's OracleVS-based ingestion and
retrieval paths:
  - id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY
  - text VARCHAR2(4000)
  - metadata JSON
  - embedding VECTOR(dim, FLOAT32)

Extra columns are fine as long as the OracleVS columns above exist and metadata JSON
contains the keys the app relies on (especially metadata.source_url for source
management and citations).

langchain-oracledb's built-in table DDL uses text CLOB. For this app's default
4000-character chunks, VARCHAR2(4000) is still OracleVS-compatible and avoids
Oracle CLOB locator failures when vector top-k SQL selects text in the same query.

The embedding dimension must match the embedding model used for both ingestion and
query. If --embedding-dim is omitted, the script resolves the dimension by making a
live embedding call with the configured model. Use --embedding-dim explicitly for
offline or dry-run workflows.

Run from project root:
  uv run scripts/create_rag_table.py [--table NAME] [--embedding-dim N] [--dry-run] [--yes]
  uv run scripts/create_rag_table.py [--table NAME] --yes --drop-existing  # recreate with new dim
  uv run scripts/create_rag_table.py [--table NAME] --yes --create-index    # add HNSW vector index
"""

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import oracledb
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import create_index
from langchain_oracledb.vectorstores.utils import DistanceStrategy

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from api.settings import get_settings
from src.rag_agent.infrastructure.oci_models import get_embedding_model


def get_table_ddl(table_name: str, embedding_dim: int) -> str:
    """Build CREATE TABLE DDL for OracleVS-compatible table."""
    table_name_upper = table_name.upper()
    ddl_body = ", ".join(
        [
            "id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
            "text VARCHAR2(4000)",
            "metadata JSON",
            f"embedding VECTOR({embedding_dim}, FLOAT32)",
        ]
    )
    return f"CREATE TABLE {table_name_upper} ({ddl_body})"


def get_embedding_dimension_from_model() -> int:
    """Resolve embedding dimension from the app's embedding model (config)."""
    embed_model_type = getattr(get_settings(), "EMBED_MODEL_TYPE", "OCI")
    embed_model = get_embedding_model(embed_model_type)
    vec = embed_model.embed_query("test")
    return len(vec)


def drop_table(conn: oracledb.Connection, table_name: str) -> bool:
    """Drop the table if it exists. Returns True if dropped, False if not found."""
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
        print(f"Table {table_name_upper} dropped.")
        return True
    finally:
        cursor.close()


def create_table(
    conn: oracledb.Connection,
    table_name: str,
    embedding_dim: int,
    drop_existing: bool = False,
) -> None:
    """Create the table. If drop_existing, drop first (e.g. to change vector dimension)."""
    cursor = conn.cursor()
    try:
        table_name_upper = table_name.upper()
        if drop_existing:
            drop_table(conn, table_name_upper)
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE UPPER(table_name) = UPPER(:1)",
                [table_name_upper],
            )
            (exists,) = cursor.fetchone()
            if exists:
                print(f"Table {table_name_upper} already exists; no change.")
                return
        ddl = get_table_ddl(table_name_upper, embedding_dim)
        cursor.execute(ddl)
        conn.commit()
        print(f"Table {table_name_upper} created successfully (embedding_dim={embedding_dim}).")
    finally:
        cursor.close()


def _vector_index_exists(conn: oracledb.Connection, index_name: str) -> bool:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM user_indexes WHERE UPPER(index_name) = UPPER(:1)",
            [index_name],
        )
        (exists,) = cursor.fetchone()
        return bool(exists)
    finally:
        cursor.close()


def create_vector_index(
    conn: oracledb.Connection,
    table_name: str,
    index_name: str,
    index_type: str = "HNSW",
) -> None:
    """Create an OracleVS vector index using langchain-oracledb helpers."""
    table_name_upper = table_name.upper()
    index_name_upper = index_name.upper()
    index_type_upper = index_type.upper()
    if _vector_index_exists(conn, index_name_upper):
        print(f"Vector index {index_name_upper} already exists; no change.")
        return
    embed_model_type = getattr(get_settings(), "EMBED_MODEL_TYPE", "OCI")
    embeddings = get_embedding_model(embed_model_type)
    vector_store = OracleVS(
        client=conn,
        embedding_function=embeddings,
        table_name=table_name_upper,
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_index(
        conn,
        vector_store,
        params={"idx_name": index_name_upper, "idx_type": index_type_upper},
    )
    print(f"Vector index {index_name_upper} created on {table_name_upper} ({index_type_upper}).")


def main() -> int:
    """Run create-table: parse args, resolve embedding dim, create or dry-run."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create RAG vector store table (OracleVS schema)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/create_rag_table.py --dry-run
  uv run scripts/create_rag_table.py --embedding-dim 1536 --yes
  uv run scripts/create_rag_table.py --table RAG_KNOWLEDGE_BASE --yes
  uv run scripts/create_rag_table.py --table RAG_KNOWLEDGE_BASE --yes --drop-existing
  uv run scripts/create_rag_table.py --table RAG_KNOWLEDGE_BASE --yes --create-index
        """,
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the table if it exists before creating (use when changing embedding dimension)",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table name (default: config.DEFAULT_COLLECTION)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: from config embedding model). "
        "Common: 1536 (OCI Cohere embed-v4.0), 384/512/768/1024/2048 (NVIDIA NIM). Oracle max: 65535.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL only, do not connect or create",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm table creation (required to run without --dry-run)",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create an OracleVS vector index after table creation/check.",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Vector index name (default: <table>_HNSW_IDX).",
    )
    parser.add_argument(
        "--index-type",
        default="HNSW",
        choices=["HNSW", "IVF"],
        help="Oracle vector index type for --create-index.",
    )
    args = parser.parse_args()

    table_name = (
        args.table or getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")
    ).strip()
    if not table_name:
        print("Error: table name is required.", file=sys.stderr)
        return 1

    if args.embedding_dim is not None:
        if args.embedding_dim < 1:
            print("Error: --embedding-dim must be >= 1.", file=sys.stderr)
            return 1
        embedding_dim = args.embedding_dim
    else:
        try:
            embedding_dim = get_embedding_dimension_from_model()
            print(f"Resolved embedding dimension from model: {embedding_dim}")
        except Exception as e:
            print(
                f"Error: could not get embedding dimension from model: {e}. "
                "Use --embedding-dim explicitly.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print(get_table_ddl(table_name, embedding_dim))
        if args.create_index:
            index_name = args.index_name or f"{table_name}_HNSW_IDX"
            print(
                "OracleVS vector index: "
                f"idx_name={index_name.upper()}, idx_type={args.index_type.upper()}"
            )
        return 0

    if not args.yes:
        print("Error: --yes is required to create the table (or use --dry-run).")
        return 1

    connect_args = cast(Mapping[str, str | None], get_settings().CONNECT_ARGS)
    conn = oracledb.connect(**connect_args)
    try:
        create_table(conn, table_name, embedding_dim, drop_existing=args.drop_existing)
        if args.create_index:
            index_name = args.index_name or f"{table_name}_HNSW_IDX"
            create_vector_index(
                conn,
                table_name=table_name,
                index_name=index_name,
                index_type=args.index_type,
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

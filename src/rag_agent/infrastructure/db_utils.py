"""Database utilities: direct connection and connection pool for the API."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import oracledb

from api.settings import get_settings

# Lazy-initialized connection pool for API (vector search, etc.)
_pool: oracledb.ConnectionPool | None = None

# Pool sizing: min connections kept open, max total
POOL_MIN = 2
POOL_MAX = 10


def _wallet_dir() -> str | None:
    """Wallet dir: env VECTOR_WALLET_DIR, else project local-config/wallet if it exists (local + Docker same dir)."""
    env_dir = os.environ.get("VECTOR_WALLET_DIR")
    if env_dir:
        return env_dir
    project_root = Path(__file__).resolve().parents[3]
    local_wallet = project_root / "local-config" / "wallet"
    if local_wallet.is_dir():
        return str(local_wallet)
    return None


def _connect_args() -> dict[str, str | None]:
    """CONNECT_ARGS from config, with wallet dir from env or project local-config/wallet when present."""
    connect_args = cast(Mapping[str, str | None], get_settings().CONNECT_ARGS)
    kwargs = dict(connect_args)
    wallet_dir = _wallet_dir()
    if wallet_dir:
        kwargs["config_dir"] = wallet_dir
        kwargs["wallet_location"] = wallet_dir
    return kwargs


def _get_pool() -> oracledb.ConnectionPool:
    """Create and return the process-wide connection pool (lazy init)."""
    global _pool
    if _pool is None:
        kwargs = _connect_args()
        timeout_sec = getattr(get_settings(), "DB_TCP_CONNECT_TIMEOUT", 5)
        # Pass explicitly so driver respects it (fail fast when DB is stopped/unreachable)
        _pool = oracledb.create_pool(
            user=kwargs.pop("user"),
            password=kwargs.pop("password"),
            dsn=kwargs.pop("dsn"),
            min=POOL_MIN,
            max=POOL_MAX,
            tcp_connect_timeout=float(timeout_sec),
            retry_count=0,
            **kwargs,
        )
    return _pool


def get_pooled_connection() -> AbstractContextManager[oracledb.Connection]:
    """
    Return a connection from the pool (context manager).
    Use in API path (e.g. vector search) to avoid per-request connection setup.
    Usage: with get_pooled_connection() as conn: ...
    """
    return _get_pool().acquire()


def close_pool(force: bool = True) -> None:
    """
    Close the process-wide connection pool. Call at application shutdown.
    If force=True, closes even when connections are still checked out.
    """
    global _pool
    if _pool is not None:
        try:
            _pool.close(force=force)
        finally:
            _pool = None


def get_connection() -> oracledb.Connection:
    """
    Get a direct connection to the DB (no pool).
    Use for one-off scripts; for request handlers prefer get_pooled_connection().
    """
    return oracledb.connect(**_connect_args())


def list_collections() -> list[str]:
    """
    Return a list of all collections (tables) with a type vector in the schema in use.
    """

    query = """
                SELECT DISTINCT table_name
                FROM user_tab_columns
                WHERE data_type = 'VECTOR'
                ORDER by table_name ASC
                """
    _collections = []
    with get_pooled_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            for row in rows:
                _collections.append(row[0])

    return sorted(_collections)


def list_sources_in_collection(collection_name: str) -> list[tuple[str | None, int]]:
    """
    Return distinct document sources in the collection from metadata, with chunk counts.
    Prefers the oracle_web_embeddings convention (`source_url`) and falls back to local ingestion keys.
    Returns a sorted list of (source, chunk_count).
    """
    validated_collection_name = _validate_collection_name(collection_name)
    with get_pooled_connection() as conn:
        with conn.cursor() as cursor:
            metadata_col = _get_metadata_column(cursor, validated_collection_name)
            if metadata_col is None:
                return []

            source_expr = (
                f"COALESCE(json_value({metadata_col}, '$.source_url'), "
                f"json_value({metadata_col}, '$.source'), "
                f"json_value({metadata_col}, '$.file_name'), "
                f"json_value({metadata_col}, '$.file_path'))"
            )
            query = f"""
                SELECT DISTINCT {source_expr} AS books,
                count(*) as n_chunks
                FROM {validated_collection_name}
                group by books
                ORDER by books ASC
                """
            cursor.execute(query)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append((row[0], row[1]))

    return sorted(result)


def delete_source_from_collection(collection_name: str, source: str) -> int:
    """Delete all chunks for a source from the selected collection and return row count."""
    validated_collection_name = _validate_collection_name(collection_name)
    with get_pooled_connection() as conn:
        with conn.cursor() as cursor:
            metadata_col = _get_metadata_column(cursor, validated_collection_name)
            if metadata_col is None:
                return 0
            source_expr = (
                f"COALESCE(json_value({metadata_col}, '$.source_url'), "
                f"json_value({metadata_col}, '$.source'), "
                f"json_value({metadata_col}, '$.file_name'), "
                f"json_value({metadata_col}, '$.file_path'))"
            )
            query = f"""
                DELETE FROM {validated_collection_name}
                WHERE {source_expr} = :source
            """
            cursor.execute(query, source=source)
            deleted_rows = cursor.rowcount or 0
        conn.commit()
    return deleted_rows


def _validate_collection_name(collection_name: str) -> str:
    """Allow only simple Oracle table identifiers for collection names."""
    normalized = collection_name.strip()
    if not normalized:
        raise ValueError("collection_name is required")
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$#")
    if any(char not in allowed_chars for char in normalized):
        raise ValueError("Invalid collection name")
    return normalized


def _get_metadata_column(cursor: oracledb.Cursor, collection_name: str) -> str | None:
    cursor.execute(
        """
        SELECT column_name FROM user_tab_columns
        WHERE UPPER(table_name) = UPPER(:1)
        AND UPPER(column_name) IN ('VMETADATA', 'METADATA')
    """,
        [collection_name],
    )
    metadata_cols = [row[0] for row in cursor.fetchall()]
    if not metadata_cols:
        return None
    return "VMETADATA" if "VMETADATA" in metadata_cols else metadata_cols[0]

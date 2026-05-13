"""Document management endpoints for the runtime API surface."""

import asyncio
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api.settings import get_settings
from src.rag_agent.infrastructure.db_utils import (
    delete_source_from_collection,
    list_sources_in_collection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

try:
    from src.rag_agent.ingestion import process_file_paths as _process_file_paths

    process_file_paths: (
        Callable[[list[str | Path], str | None], tuple[bool, int, str | None]] | None
    ) = _process_file_paths
except ImportError:
    process_file_paths = None


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(default=[]),
    collection_name: str | None = Form(default=None),
) -> dict[str, object]:
    if process_file_paths is None:
        return {"error": "Document upload not available", "chunks_added": 0}
    if not files:
        return {"error": "No files provided", "chunks_added": 0}

    allowed = {"pdf", "html", "htm", "txt", "md", "markdown"}
    paths: list[Path] = []
    temp_dir = tempfile.mkdtemp()
    try:
        for uploaded_file in files:
            if not uploaded_file.filename:
                continue
            ext = Path(uploaded_file.filename).suffix.lower().lstrip(".")
            if ext not in allowed:
                continue
            path = Path(temp_dir) / (uploaded_file.filename or "upload")
            content = await uploaded_file.read()
            path.write_bytes(content)
            paths.append(path)

        if not paths:
            return {
                "error": "No supported files (pdf, html, htm, txt, md)",
                "chunks_added": 0,
            }

        table_name = (collection_name and collection_name.strip()) or getattr(
            get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE"
        )
        assert process_file_paths is not None
        success, num_chunks, err = await asyncio.to_thread(
            process_file_paths,
            cast(list[str | Path], paths),
            table_name,
        )
        if success:
            return {
                "chunks_added": num_chunks,
                "files_processed": len(paths),
                "collection": table_name,
            }

        logger.error("Upload failed: %s", err)
        return {"error": err or "Processing failed", "chunks_added": 0}
    finally:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Cleanup unlink %s: %s", path, exc)
        try:
            Path(temp_dir).rmdir()
        except OSError as exc:
            logger.debug("Cleanup rmdir %s: %s", temp_dir, exc)


def _resolve_collection_name(collection_name: str | None) -> str:
    if collection_name:
        stripped_name = collection_name.strip()
        if stripped_name:
            return stripped_name

    default_collection = getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")
    return default_collection if isinstance(default_collection, str) else "RAG_KNOWLEDGE_BASE"


@router.get("/sources")
async def list_document_sources(collection_name: str | None = Query(default=None)) -> dict[str, object]:
    table_name = _resolve_collection_name(collection_name)
    rows = await asyncio.to_thread(list_sources_in_collection, table_name)
    sources = [
        {"source": source, "chunk_count": chunk_count}
        for source, chunk_count in rows
        if isinstance(source, str) and source.strip()
    ]
    return {"collection": table_name, "sources": sources}


@router.delete("/source")
async def delete_document_source(
    source: str | None = Query(default=None),
    collection_name: str | None = Query(default=None),
) -> dict[str, object]:
    if source is None or not source.strip():
        raise HTTPException(status_code=400, detail="source is required")

    table_name = _resolve_collection_name(collection_name)
    deleted_chunks = await asyncio.to_thread(
        delete_source_from_collection,
        table_name,
        source.strip(),
    )
    return {
        "collection": table_name,
        "source": source.strip(),
        "deleted_chunks": deleted_chunks,
    }

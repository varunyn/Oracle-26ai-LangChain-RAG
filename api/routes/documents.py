"""Document management endpoints for the runtime API surface."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api.services.document_ingestion_jobs import (
    create_ingestion_job,
    get_ingestion_job,
    run_ingestion_job_sync,
)
from api.settings import get_settings
from src.rag_agent.infrastructure.db_utils import (
    delete_source_from_collection,
    list_sources_in_collection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

try:
    from src.rag_agent.ingestion import process_file_paths as _process_file_paths

    process_file_paths: Callable[..., tuple[bool, int, str | None]] | None = _process_file_paths
except ImportError:
    process_file_paths = None


def _schedule_ingestion_job(job_id: str, paths: list[Path], table_name: str) -> None:
    assert process_file_paths is not None
    asyncio.create_task(
        asyncio.to_thread(
            run_ingestion_job_sync,
            job_id=job_id,
            paths=paths,
            collection_name=table_name,
            process_file_paths=process_file_paths,
        )
    )


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
    accepted_files: list[tuple[str, bytes]] = []
    for uploaded_file in files:
        if not uploaded_file.filename:
            continue
        filename = Path(uploaded_file.filename).name
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in allowed:
            continue
        content = await uploaded_file.read()
        accepted_files.append((filename, content))

    if not accepted_files:
        return {
            "error": "No supported files (pdf, html, htm, txt, md)",
            "chunks_added": 0,
        }

    table_name = _resolve_collection_name(collection_name)
    job, paths = create_ingestion_job(collection_name=table_name, files=accepted_files)
    _schedule_ingestion_job(str(job["job_id"]), paths, table_name)

    return {
        **job,
        "chunks_added": 0,
        "files_processed": 0,
    }


@router.get("/jobs/{job_id}")
async def get_document_ingestion_job(job_id: str) -> dict[str, object]:
    job = get_ingestion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ingestion job not found")
    return job


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

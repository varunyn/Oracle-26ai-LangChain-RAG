"""Durable document ingestion job tracking."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

ProcessFilePaths = Callable[..., tuple[bool, int, str | None]]

_LOCAL_DATA_DIR = Path(".local-data")
_JOBS_DIR = _LOCAL_DATA_DIR / "ingestion-jobs"
_UPLOADS_DIR = _LOCAL_DATA_DIR / "ingestion-uploads"
_TERMINAL_JOB_STATUSES = {"completed", "failed", "interrupted"}
_ORPHANED_JOB_GRACE_SECONDS = 15
_ACTIVE_JOBS: set[str] = set()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_dirs() -> None:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    _ensure_dirs()
    path = _job_path(str(job["job_id"]))
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(job, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read ingestion job %s: %s", job_id, exc)
        return None


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    public = dict(job)
    public["files"] = [
        {key: value for key, value in file_item.items() if key != "path"}
        for file_item in list(job.get("files", []))
        if isinstance(file_item, dict)
    ]
    return public


def _is_stale_orphan(job: dict[str, Any]) -> bool:
    updated_at = job.get("updated_at")
    if not isinstance(updated_at, str):
        return True
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return True
    age_seconds = (datetime.now(UTC) - updated).total_seconds()
    return age_seconds >= _ORPHANED_JOB_GRACE_SECONDS


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name or "upload"


def _file_for_path(job: dict[str, Any], file_path: str | Path) -> dict[str, Any] | None:
    target = str(Path(file_path))
    target_name = Path(file_path).name
    for file_item in job.get("files", []):
        if not isinstance(file_item, dict):
            continue
        if file_item.get("path") == target or file_item.get("name") == target_name:
            return file_item
    return None


def create_ingestion_job(
    *,
    collection_name: str,
    files: list[tuple[str, bytes]],
) -> tuple[dict[str, Any], list[Path]]:
    _ensure_dirs()
    job_id = uuid.uuid4().hex
    upload_dir = _UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    job_files: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index, (filename, content) in enumerate(files):
        safe_name = _sanitize_filename(filename)
        path = upload_dir / f"{index:03d}-{safe_name}"
        path.write_bytes(content)
        paths.append(path)
        job_files.append(
            {
                "file_id": uuid.uuid4().hex,
                "name": safe_name,
                "status": "queued",
                "chunks_added": 0,
                "message": None,
                "path": str(path),
                "started_at": None,
                "completed_at": None,
            }
        )

    now = _now_iso()
    job: dict[str, Any] = {
        "job_id": job_id,
        "collection": collection_name,
        "status": "queued",
        "chunks_added": 0,
        "files_processed": 0,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "files": job_files,
    }
    _write_job(job)
    return _public_job(job), paths


def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    job = _read_job(job_id)
    if job is None:
        return None

    status = str(job.get("status", ""))
    if (
        status not in _TERMINAL_JOB_STATUSES
        and job_id not in _ACTIVE_JOBS
        and _is_stale_orphan(job)
    ):
        _mark_interrupted(job, "Backend restarted before ingestion completed.")
        _write_job(job)

    return _public_job(job)


def _mark_interrupted(job: dict[str, Any], message: str) -> None:
    now = _now_iso()
    job["status"] = "interrupted"
    job["error"] = message
    job["updated_at"] = now
    for file_item in job.get("files", []):
        if isinstance(file_item, dict) and file_item.get("status") not in {
            "indexed",
            "failed",
        }:
            file_item["status"] = "interrupted"
            file_item["message"] = message
            file_item["completed_at"] = now


def _update_file_status(
    job_id: str,
    file_path: str | Path,
    status: str,
    payload: dict[str, Any] | None = None,
) -> None:
    job = _read_job(job_id)
    if job is None:
        return

    now = _now_iso()
    job["updated_at"] = now
    job["status"] = "running"
    file_item = _file_for_path(job, file_path)
    if file_item is None:
        return

    file_item["status"] = status
    if file_item.get("started_at") is None and status != "queued":
        file_item["started_at"] = now
    if payload:
        if "chunks_added" in payload:
            file_item["chunks_added"] = int(payload["chunks_added"])
        if "message" in payload:
            file_item["message"] = payload["message"]
    if status in {"indexed", "failed", "interrupted"}:
        file_item["completed_at"] = now

    _write_job(job)


def _complete_job(job_id: str, success: bool, chunks_added: int, error: str | None) -> None:
    job = _read_job(job_id)
    if job is None:
        return

    now = _now_iso()
    indexed_files = [
        file_item
        for file_item in job.get("files", [])
        if isinstance(file_item, dict) and file_item.get("status") == "indexed"
    ]
    job["status"] = "completed" if success else "failed"
    job["chunks_added"] = chunks_added
    job["files_processed"] = len(indexed_files)
    job["error"] = error
    job["updated_at"] = now
    for file_item in job.get("files", []):
        if isinstance(file_item, dict) and file_item.get("status") in {"queued", "running"}:
            file_item["status"] = "failed"
            file_item["message"] = error or "Processing failed."
            file_item["completed_at"] = now

    _write_job(job)


def run_ingestion_job_sync(
    *,
    job_id: str,
    paths: list[Path],
    collection_name: str,
    process_file_paths: ProcessFilePaths,
) -> None:
    _ACTIVE_JOBS.add(job_id)
    try:
        job = _read_job(job_id)
        if job is not None:
            job["status"] = "running"
            job["updated_at"] = _now_iso()
            _write_job(job)

        def progress_callback(
            file_path: str | Path,
            status: str,
            payload: dict[str, Any] | None = None,
        ) -> None:
            _update_file_status(job_id, file_path, status, payload)

        supports_progress_callback = "progress_callback" in signature(process_file_paths).parameters
        if supports_progress_callback:
            success, chunks_added, error = process_file_paths(
                paths,
                collection_name,
                progress_callback=progress_callback,
            )
        else:
            success, chunks_added, error = process_file_paths(paths, collection_name)

        _complete_job(job_id, success, chunks_added, error)
    except Exception as exc:
        logger.exception("Ingestion job %s failed", job_id)
        _complete_job(job_id, False, 0, str(exc))
    finally:
        _ACTIVE_JOBS.discard(job_id)
        upload_dir = _UPLOADS_DIR / job_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)

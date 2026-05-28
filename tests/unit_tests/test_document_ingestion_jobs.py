from __future__ import annotations

from pathlib import Path

from api.services import document_ingestion_jobs as jobs


def test_ingestion_job_runner_records_progress(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(jobs, "_LOCAL_DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_JOBS_DIR", tmp_path / "ingestion-jobs")
    monkeypatch.setattr(jobs, "_UPLOADS_DIR", tmp_path / "ingestion-uploads")

    job, paths = jobs.create_ingestion_job(
        collection_name="MY_COLLECTION",
        files=[("doc.txt", b"hello")],
    )

    def fake_process_file_paths(
        file_paths: list[Path],
        table_name: str,
        *,
        progress_callback,
    ) -> tuple[bool, int, str | None]:
        progress_callback(file_paths[0], "parsing", None)
        progress_callback(file_paths[0], "embedding", None)
        progress_callback(file_paths[0], "indexed", {"chunks_added": 5})
        return True, 5, None

    jobs.run_ingestion_job_sync(
        job_id=str(job["job_id"]),
        paths=paths,
        collection_name="MY_COLLECTION",
        process_file_paths=fake_process_file_paths,
    )

    stored_job = jobs.get_ingestion_job(str(job["job_id"]))

    assert stored_job is not None
    assert stored_job["status"] == "completed"
    assert stored_job["chunks_added"] == 5
    assert stored_job["files_processed"] == 1
    assert stored_job["files"][0]["status"] == "indexed"
    assert stored_job["files"][0]["chunks_added"] == 5
    assert "path" not in stored_job["files"][0]


def test_ingestion_job_get_marks_orphaned_job_interrupted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(jobs, "_LOCAL_DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_JOBS_DIR", tmp_path / "ingestion-jobs")
    monkeypatch.setattr(jobs, "_UPLOADS_DIR", tmp_path / "ingestion-uploads")
    monkeypatch.setattr(jobs, "_ORPHANED_JOB_GRACE_SECONDS", 0)

    job, _ = jobs.create_ingestion_job(
        collection_name="MY_COLLECTION",
        files=[("doc.txt", b"hello")],
    )

    stored_job = jobs.get_ingestion_job(str(job["job_id"]))

    assert stored_job is not None
    assert stored_job["status"] == "interrupted"
    assert stored_job["error"] == "Backend restarted before ingestion completed."
    assert stored_job["files"][0]["status"] == "interrupted"

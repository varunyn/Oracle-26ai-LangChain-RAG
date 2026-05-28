from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import documents

app = FastAPI()
app.include_router(documents.router)
client = TestClient(app)


def test_upload_documents_returns_error_when_no_files_provided() -> None:
    response = client.post("/api/documents/upload")

    assert response.status_code == 200
    assert response.json() == {"error": "No files provided", "chunks_added": 0}


def test_upload_documents_rejects_unsupported_files() -> None:
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("notes.csv", BytesIO(b"a,b,c"), "text/csv"))],
    )

    assert response.status_code == 200
    assert response.json() == {
        "error": "No supported files (pdf, html, htm, txt, md)",
        "chunks_added": 0,
    }


def test_upload_documents_processes_supported_files(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_ingestion_job(
        *, collection_name: str, files: list[tuple[str, bytes]]
    ) -> tuple[dict[str, object], list[Path]]:
        captured["collection_name"] = collection_name
        captured["files"] = files
        return (
            {
                "job_id": "job-1",
                "collection": collection_name,
                "status": "queued",
                "chunks_added": 0,
                "files_processed": 0,
                "files": [
                    {
                        "file_id": "file-1",
                        "name": "doc.txt",
                        "status": "queued",
                        "chunks_added": 0,
                    }
                ],
            },
            [Path("/tmp/doc.txt")],
        )

    def fake_schedule_ingestion_job(job_id: str, paths: list[Path], table_name: str) -> None:
        captured["job_id"] = job_id
        captured["paths"] = paths
        captured["table_name"] = table_name

    monkeypatch.setattr(documents, "create_ingestion_job", fake_create_ingestion_job)
    monkeypatch.setattr(documents, "_schedule_ingestion_job", fake_schedule_ingestion_job)

    response = client.post(
        "/api/documents/upload",
        data={"collection_name": "MY_COLLECTION"},
        files=[("files", ("doc.txt", BytesIO(b"hello"), "text/plain"))],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-1"
    assert data["status"] == "queued"
    assert data["chunks_added"] == 0
    assert data["files_processed"] == 0
    assert captured["table_name"] == "MY_COLLECTION"
    assert captured["collection_name"] == "MY_COLLECTION"
    assert captured["files"] == [("doc.txt", b"hello")]


def test_get_document_ingestion_job_returns_job(monkeypatch) -> None:
    def fake_get_ingestion_job(job_id: str) -> dict[str, object] | None:
        assert job_id == "job-1"
        return {"job_id": job_id, "status": "failed", "error": "broken"}

    monkeypatch.setattr(documents, "get_ingestion_job", fake_get_ingestion_job)

    response = client.get("/api/documents/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-1", "status": "failed", "error": "broken"}


def test_get_document_ingestion_job_returns_404_for_missing_job(monkeypatch) -> None:
    monkeypatch.setattr(documents, "get_ingestion_job", lambda job_id: None)

    response = client.get("/api/documents/jobs/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "ingestion job not found"}


def test_list_document_sources_returns_grouped_sources(monkeypatch) -> None:
    def fake_list_sources_in_collection(collection_name: str) -> list[tuple[str | None, int]]:
        assert collection_name == "MY_COLLECTION"
        return [("https://example.com/a", 3), (None, 2), ("notes.md", 1)]

    monkeypatch.setattr(documents, "list_sources_in_collection", fake_list_sources_in_collection)

    response = client.get("/api/documents/sources?collection_name=MY_COLLECTION")

    assert response.status_code == 200
    assert response.json() == {
        "collection": "MY_COLLECTION",
        "sources": [
            {"source": "https://example.com/a", "chunk_count": 3},
            {"source": "notes.md", "chunk_count": 1},
        ],
    }


def test_list_document_sources_uses_default_collection(monkeypatch) -> None:
    monkeypatch.setattr(documents, "list_sources_in_collection", lambda name: [])

    response = client.get("/api/documents/sources")

    assert response.status_code == 200
    assert response.json()["collection"] == documents.get_settings().DEFAULT_COLLECTION


def test_delete_document_source_returns_deleted_chunk_count(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_delete_source_from_collection(collection_name: str, source: str) -> int:
        captured["collection_name"] = collection_name
        captured["source"] = source
        return 4

    monkeypatch.setattr(
        documents, "delete_source_from_collection", fake_delete_source_from_collection
    )

    response = client.delete(
        "/api/documents/source?collection_name=MY_COLLECTION&source=https://example.com/a"
    )

    assert response.status_code == 200
    assert response.json() == {
        "collection": "MY_COLLECTION",
        "source": "https://example.com/a",
        "deleted_chunks": 4,
    }
    assert captured == {
        "collection_name": "MY_COLLECTION",
        "source": "https://example.com/a",
    }


def test_delete_document_source_requires_source() -> None:
    response = client.delete("/api/documents/source")

    assert response.status_code == 400
    assert response.json() == {"detail": "source is required"}

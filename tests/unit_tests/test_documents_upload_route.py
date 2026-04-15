from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rag_agent.runtime import documents

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

    def fake_process_file_paths(
        file_paths: list[str | Path], table_name: str | None = None
    ) -> tuple[bool, int, str | None]:
        captured["file_paths"] = file_paths
        captured["table_name"] = table_name
        return True, 7, None

    monkeypatch.setattr(documents, "process_file_paths", fake_process_file_paths)

    response = client.post(
        "/api/documents/upload",
        data={"collection_name": "MY_COLLECTION"},
        files=[("files", ("doc.txt", BytesIO(b"hello"), "text/plain"))],
    )

    assert response.status_code == 200
    assert response.json() == {
        "chunks_added": 7,
        "files_processed": 1,
        "collection": "MY_COLLECTION",
    }
    assert captured["table_name"] == "MY_COLLECTION"
    file_paths = captured["file_paths"]
    assert isinstance(file_paths, list)
    assert len(file_paths) == 1
    assert Path(file_paths[0]).name == "doc.txt"


def test_upload_documents_surfaces_ingestion_failures(monkeypatch) -> None:
    def fake_process_file_paths(
        file_paths: list[str | Path], table_name: str | None = None
    ) -> tuple[bool, int, str | None]:
        return False, 0, "broken"

    monkeypatch.setattr(documents, "process_file_paths", fake_process_file_paths)

    response = client.post(
        "/api/documents/upload",
        files=[("files", ("doc.txt", BytesIO(b"hello"), "text/plain"))],
    )

    assert response.status_code == 200
    assert response.json() == {"error": "broken", "chunks_added": 0}


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

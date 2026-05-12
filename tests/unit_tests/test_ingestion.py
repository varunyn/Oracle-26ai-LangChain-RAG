from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.rag_agent import ingestion


class FakeSplitter:
    def split_text(self, text: str) -> list[str]:
        return [text]


class FakeConnection:
    def __init__(self) -> None:
        self.autocommit = False
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeOracleVS:
    last_init: dict[str, object] | None = None
    last_add_documents: dict[str, object] | None = None

    def __init__(self, client, embedding_function, table_name, distance_strategy):
        type(self).last_init = {
            "client": client,
            "embedding_function": embedding_function,
            "table_name": table_name,
            "distance_strategy": distance_strategy,
        }

    def add_documents(self, documents, **kwargs):
        type(self).last_add_documents = {
            "documents": documents,
            "kwargs": kwargs,
        }
        return ["doc-1#chunk-0", "doc-2#chunk-0"]


def test_process_file_paths_returns_error_when_no_documents(monkeypatch) -> None:
    monkeypatch.setattr(ingestion, "load_documents_from_files", lambda files: [])

    success, num_chunks, error = ingestion.process_file_paths(["missing.txt"])

    assert success is False
    assert num_chunks == 0
    assert error == "No documents loaded (unsupported type or empty)."


def test_process_file_paths_uses_default_table_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion,
        "load_documents_from_files",
        lambda files: [SimpleNamespace(page_content="hello", metadata={})],
    )
    monkeypatch.setattr(ingestion, "_split_and_store", lambda docs, table_name: 5)

    success, num_chunks, error = ingestion.process_file_paths(["doc.txt"])

    assert success is True
    assert num_chunks == 5
    assert error is None


def test_process_file_paths_returns_error_when_documents_have_no_text(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion,
        "load_documents_from_files",
        lambda files: [
            SimpleNamespace(page_content="   ", metadata={"source": "scan.pdf"}),
            SimpleNamespace(page_content="", metadata={"source": "scan.pdf"}),
        ],
    )

    success, num_chunks, error = ingestion.process_file_paths(["scan.pdf"])

    assert success is False
    assert num_chunks == 0
    assert error == "No text content could be extracted from the uploaded documents."


def test_process_file_paths_returns_error_when_no_chunks_inserted(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion,
        "load_documents_from_files",
        lambda files: [SimpleNamespace(page_content="hello", metadata={"source": "doc.txt"})],
    )
    monkeypatch.setattr(ingestion, "_split_and_store", lambda docs, table_name: 0)

    success, num_chunks, error = ingestion.process_file_paths(["doc.txt"])

    assert success is False
    assert num_chunks == 0
    assert error == "No chunks were inserted into the vector store."


def test_split_and_store_delegates_chunking_and_ids_to_oraclevs(monkeypatch) -> None:
    docs = [
        SimpleNamespace(page_content="alpha", metadata={"source": "a.txt"}),
        SimpleNamespace(page_content="beta", metadata={"source": "b.txt"}),
    ]
    fake_connection = FakeConnection()
    recorded_splitter_kwargs: dict[str, object] = {}

    def make_fake_splitter(**kwargs):
        recorded_splitter_kwargs.update(kwargs)
        return FakeSplitter()

    monkeypatch.setattr(
        ingestion,
        "RecursiveCharacterTextSplitter",
        make_fake_splitter,
    )
    monkeypatch.setattr(
        ingestion,
        "get_settings",
        lambda: SimpleNamespace(
            CHUNK_SIZE=1234,
            CHUNK_OVERLAP=234,
            CONNECT_ARGS={},
            EMBED_MODEL_TYPE="OCI",
        ),
    )
    monkeypatch.setattr(ingestion.oracledb, "connect", lambda **kwargs: fake_connection)
    monkeypatch.setattr(ingestion, "get_embedding_model", lambda model_type: "embeddings")
    monkeypatch.setattr(ingestion, "OracleVS", FakeOracleVS)

    count = ingestion._split_and_store(docs, table_name="COLL_A")

    assert count == 2
    assert docs[0].id.startswith("a.txt::0::")
    assert docs[1].id.startswith("b.txt::1::")
    assert recorded_splitter_kwargs["chunk_size"] == 1234
    assert recorded_splitter_kwargs["chunk_overlap"] == 234
    assert recorded_splitter_kwargs["length_function"] is len
    assert recorded_splitter_kwargs["separators"] == ["\n\n", "\n", " ", ""]
    assert fake_connection.autocommit is True
    assert fake_connection.closed is True
    assert FakeOracleVS.last_init is not None
    assert FakeOracleVS.last_init["table_name"] == "COLL_A"
    assert FakeOracleVS.last_add_documents is not None
    assert FakeOracleVS.last_add_documents["documents"] == docs
    assert FakeOracleVS.last_add_documents["kwargs"]["text_splitter"].__class__ is FakeSplitter


def test_convert_file_with_docling_exports_markdown(monkeypatch, tmp_path) -> None:
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")
    captured: dict[str, object] = {}

    class FakeDoclingDocument:
        def export_to_markdown(self) -> str:
            return "# Converted\n\nBody text"

    class FakeConverter:
        def convert(self, path: Path):
            captured["path"] = path
            return SimpleNamespace(document=FakeDoclingDocument())

    monkeypatch.setattr(ingestion, "_build_docling_converter", lambda: FakeConverter())

    markdown = ingestion._convert_file_with_docling(source_file)

    assert markdown == "# Converted\n\nBody text"
    assert captured["path"] == source_file


def test_load_document_with_docling_sets_metadata(monkeypatch, tmp_path) -> None:
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")

    monkeypatch.setattr(ingestion, "_convert_file_with_docling", lambda path: "# Title\n\nBody")
    monkeypatch.setattr(
        ingestion, "copy_file_to_uploaded", lambda path: "uploaded_files/doc_123.txt"
    )

    docs = ingestion.load_document_with_docling(source_file)

    assert len(docs) == 1
    assert docs[0].page_content == "# Title\n\nBody"
    assert docs[0].metadata["source"] == "doc.txt"
    assert docs[0].metadata["file_name"] == "doc.txt"
    assert docs[0].metadata["source_url"] == "uploaded_files/doc_123.txt"
    assert docs[0].metadata["source_type"] == "file"


def test_load_document_with_docling_returns_empty_when_markdown_is_blank(
    monkeypatch, tmp_path
) -> None:
    source_file = tmp_path / "doc.pdf"
    source_file.write_bytes(b"%PDF")

    monkeypatch.setattr(ingestion, "_convert_file_with_docling", lambda path: "   \n")
    monkeypatch.setattr(
        ingestion, "copy_file_to_uploaded", lambda path: "uploaded_files/doc_123.pdf"
    )

    assert ingestion.load_document_with_docling(source_file) == []

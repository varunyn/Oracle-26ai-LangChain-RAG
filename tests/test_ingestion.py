from __future__ import annotations

from types import SimpleNamespace

from src.rag_agent import ingestion


class FakeSplitter:
    def __init__(self, split_docs):
        self._split_docs = split_docs

    def split_documents(self, docs):
        return self._split_docs


class FakeConnection:
    def __init__(self) -> None:
        self.autocommit = False
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeOracleVS:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_documents(cls, documents, embedding, client, table_name, distance_strategy):
        cls.last_call = {
            "documents": documents,
            "embedding": embedding,
            "client": client,
            "table_name": table_name,
            "distance_strategy": distance_strategy,
        }


def test_process_file_paths_returns_error_when_no_documents(monkeypatch) -> None:
    monkeypatch.setattr(ingestion, "load_documents_from_files", lambda files: [])

    success, num_chunks, error = ingestion.process_file_paths(["missing.txt"])

    assert success is False
    assert num_chunks == 0
    assert error == "No documents loaded (unsupported type or empty)."


def test_process_file_paths_uses_default_table_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion, "load_documents_from_files", lambda files: [SimpleNamespace(metadata={})]
    )
    monkeypatch.setattr(ingestion, "_split_and_store", lambda docs, table_name: 5)

    success, num_chunks, error = ingestion.process_file_paths(["doc.txt"])

    assert success is True
    assert num_chunks == 5
    assert error is None


def test_split_and_store_adds_chunk_offsets_and_uses_requested_table(monkeypatch) -> None:
    split_docs = [SimpleNamespace(metadata={}), SimpleNamespace(metadata={})]
    fake_connection = FakeConnection()
    recorded_splitter_kwargs: dict[str, object] = {}

    def make_fake_splitter(**kwargs):
        recorded_splitter_kwargs.update(kwargs)
        return FakeSplitter(split_docs)

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

    count = ingestion._split_and_store([SimpleNamespace(metadata={})], table_name="COLL_A")

    assert count == 2
    assert split_docs[0].metadata["chunk_offset"] == 0
    assert split_docs[1].metadata["chunk_offset"] == 1
    assert recorded_splitter_kwargs["chunk_size"] == 1234
    assert recorded_splitter_kwargs["chunk_overlap"] == 234
    assert recorded_splitter_kwargs["length_function"] is len
    assert recorded_splitter_kwargs["separators"] == ["\n\n", "\n", " ", ""]
    assert fake_connection.autocommit is True
    assert fake_connection.closed is True
    assert FakeOracleVS.last_call is not None
    assert FakeOracleVS.last_call["table_name"] == "COLL_A"


def test_load_document_with_langchain_sets_metadata(monkeypatch, tmp_path) -> None:
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")
    loaded_doc = SimpleNamespace(metadata={})

    class FakeLoader:
        def __init__(self, path: str, encoding: str | None = None) -> None:
            self.path = path
            self.encoding = encoding

        def load(self):
            return [loaded_doc]

    monkeypatch.setattr(ingestion, "TextLoader", FakeLoader)
    monkeypatch.setattr(
        ingestion, "copy_file_to_uploaded", lambda path: "uploaded_files/doc_123.txt"
    )

    docs = ingestion.load_document_with_langchain(source_file)

    assert docs == [loaded_doc]
    assert loaded_doc.metadata["source"] == "doc.txt"
    assert loaded_doc.metadata["file_name"] == "doc.txt"
    assert loaded_doc.metadata["source_url"] == "uploaded_files/doc_123.txt"
    assert loaded_doc.metadata["source_type"] == "file"

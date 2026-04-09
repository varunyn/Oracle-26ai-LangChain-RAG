from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import oracledb
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_oracledb import OracleVS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.settings import get_settings
from src.rag_agent.infrastructure.oci_models import get_embedding_model
from src.rag_agent.utils.utils import get_console_logger

logger = get_console_logger(__name__, level="INFO")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_NAME = getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")
UPLOADED_FILES_DIR = "uploaded_files"
SUPPORTED_EXTENSIONS = {"pdf", "html", "htm", "txt", "md", "markdown"}


def get_project_root() -> Path:
    return _PROJECT_ROOT


def ensure_uploaded_files_dir() -> Path:
    uploaded_dir = get_project_root() / UPLOADED_FILES_DIR
    uploaded_dir.mkdir(exist_ok=True)
    return uploaded_dir


def copy_file_to_uploaded(file_path: str | Path) -> str:
    try:
        uploaded_dir = ensure_uploaded_files_dir()
        original_path = Path(file_path)
        file_stem = original_path.stem
        file_ext = original_path.suffix
        unique_id = str(uuid.uuid4())[:8]
        new_filename = f"{file_stem}_{unique_id}{file_ext}"
        destination = uploaded_dir / new_filename
        shutil.copy2(file_path, destination)
        relative_path = destination.relative_to(get_project_root())
        print(f"Copied file to: {relative_path}")
        return str(relative_path)
    except Exception as e:
        print(f"Warning: Could not copy file to uploaded_files: {e}")
        return f"file://{file_path}"


def load_document_with_langchain(file_path: str | Path):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        return []

    ext = path.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported file type: {ext}")
        return []

    stored_path = copy_file_to_uploaded(path)
    original_name = path.name
    base_metadata = {
        "source": original_name,
        "source_url": stored_path,
        "file_name": original_name,
        "source_type": "file",
    }

    docs = []
    try:
        if ext == "pdf":
            docs = PyPDFLoader(str(path)).load()
        elif ext in ("html", "htm"):
            docs = UnstructuredFileLoader(str(path)).load()
        elif ext in ("txt", "md", "markdown"):
            docs = TextLoader(str(path), encoding="utf-8").load()
        else:
            return []
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

    for doc in docs:
        doc.metadata.update(base_metadata)
    return docs


def load_documents_from_files(files: Sequence[str | Path]):
    all_docs = []
    for file_path in files:
        print(f"Loading: {file_path}")
        docs = load_document_with_langchain(file_path)
        all_docs.extend(docs)
    return all_docs


def load_documents_from_dir(dir_path: str | Path):
    path = Path(dir_path)
    if not path.is_dir():
        print(f"Not a directory: {path}")
        return []

    all_docs = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in path.rglob(f"*.{ext}"):
            docs = load_document_with_langchain(file_path)
            all_docs.extend(docs)
    return all_docs


def populate_from_files(files: list[str], table_name: str = DEFAULT_TABLE_NAME) -> None:
    docs = load_documents_from_files(files)
    if not docs:
        print("No documents loaded.")
        return
    _split_and_store(docs, table_name=table_name)


def populate_from_dir(dir_path: str | Path, table_name: str = DEFAULT_TABLE_NAME) -> None:
    docs = load_documents_from_dir(dir_path)
    if not docs:
        print("No documents loaded from directory.")
        return
    _split_and_store(docs, table_name=table_name)


def _split_and_store(docs, table_name: str = DEFAULT_TABLE_NAME) -> int:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = splitter.split_documents(docs)
    if not split_docs:
        print("No chunks after splitting.")
        return 0

    for i, chunk in enumerate(split_docs):
        chunk.metadata["chunk_offset"] = i

    print(f"Split into {len(split_docs)} chunks.")
    connect_args = cast(Mapping[str, str | None], settings.CONNECT_ARGS)
    conn = oracledb.connect(**connect_args)
    conn.autocommit = True
    embed_model_type = getattr(settings, "EMBED_MODEL_TYPE", "OCI")
    embeddings = get_embedding_model(embed_model_type)
    print("Storing in OracleVS (embedding and inserting)...")
    OracleVS.from_documents(
        split_docs,
        embedding=embeddings,
        client=conn,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
    )
    conn.close()
    print(f"Successfully populated {table_name} with {len(split_docs)} chunks.")
    return len(split_docs)


def process_file_paths(
    file_paths: list[str | Path], table_name: str | None = None
) -> tuple[bool, int, str | None]:
    try:
        docs = load_documents_from_files(file_paths)
        if not docs:
            return False, 0, "No documents loaded (unsupported type or empty)."
        tbl = table_name or DEFAULT_TABLE_NAME
        num_chunks = _split_and_store(docs, table_name=tbl)
        return True, num_chunks, None
    except Exception as e:
        logger.error("process_file_paths error: %s", e)
        return False, 0, str(e)

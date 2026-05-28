from __future__ import annotations

import gc
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_oracledb import OracleVS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.settings import get_settings
from src.rag_agent.infrastructure.oci_models import get_embedding_model
from src.rag_agent.utils.utils import get_console_logger

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = get_console_logger(__name__, level="INFO")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_NAME = getattr(get_settings(), "DEFAULT_COLLECTION", "RAG_KNOWLEDGE_BASE")
UPLOADED_FILES_DIR = "uploaded_files"
SUPPORTED_EXTENSIONS = {"pdf", "html", "htm", "txt", "md", "markdown"}
IngestionProgressCallback = Callable[[str | Path, str, dict[str, object] | None], None]


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


@lru_cache(maxsize=1)
def _build_docling_converter() -> DocumentConverter:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions()
    settings = get_settings()
    pdf_options.do_ocr = bool(getattr(settings, "DOCLING_DO_OCR", True))
    pdf_options.do_table_structure = False
    pdf_options.document_timeout = int(getattr(settings, "DOCLING_DOCUMENT_TIMEOUT", 90))
    pdf_options.ocr_options = RapidOcrOptions(
        lang=["english"],
        force_full_page_ocr=bool(getattr(settings, "DOCLING_FORCE_FULL_PAGE_OCR", False)),
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


def _convert_file_with_docling(file_path: str | Path) -> str:
    result = _build_docling_converter().convert(Path(file_path))
    return result.document.export_to_markdown()


def _release_docling_converter() -> None:
    _build_docling_converter.cache_clear()
    gc.collect()


def load_document_with_docling(file_path: str | Path) -> list[Document]:
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

    try:
        markdown = _convert_file_with_docling(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

    content = markdown.strip()
    if not content:
        return []

    return [Document(page_content=content, metadata=base_metadata)]


def load_document_with_langchain(file_path: str | Path) -> list[Document]:
    return load_document_with_docling(file_path)


def load_documents_from_files(files: Sequence[str | Path]):
    all_docs = []
    for file_path in files:
        try:
            print(f"Loading: {file_path}")
            docs = load_document_with_docling(file_path)
            all_docs.extend(docs)
        finally:
            _release_docling_converter()
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


def _ensure_document_ids(docs: Sequence[object]) -> None:
    """Set stable source/content IDs so OracleVS can derive stable chunk IDs."""
    for index, doc in enumerate(docs):
        if getattr(doc, "id", None):
            continue
        metadata = getattr(doc, "metadata", {}) or {}
        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("source_url")
            or "document"
        )
        page = metadata.get("page") or metadata.get("page_number") or index
        content = str(getattr(doc, "page_content", ""))
        digest = sha256(content.encode("utf-8")).hexdigest()[:16]
        setattr(doc, "id", f"{source}::{page}::{digest}")


def _has_extractable_text(docs: Sequence[object]) -> bool:
    return any(str(getattr(doc, "page_content", "")).strip() for doc in docs)


def _split_and_store(docs, table_name: str = DEFAULT_TABLE_NAME) -> int:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    _ensure_document_ids(docs)
    connect_args = cast(Mapping[str, str | None], settings.CONNECT_ARGS)
    conn = oracledb.connect(**connect_args)
    try:
        conn.autocommit = True
        embed_model_type = getattr(settings, "EMBED_MODEL_TYPE", "OCI")
        embeddings = get_embedding_model(embed_model_type)
        print("Storing in OracleVS (chunking, embedding, and inserting)...")
        vector_store = OracleVS(
            client=conn,
            embedding_function=embeddings,
            table_name=table_name,
            distance_strategy=DistanceStrategy.COSINE,
        )
        inserted_ids = vector_store.add_documents(docs, text_splitter=splitter)
    finally:
        conn.close()

    chunk_count = len(inserted_ids)
    if chunk_count == 0:
        print("No chunks inserted.")
        return 0
    print(f"Successfully populated {table_name} with {chunk_count} chunks.")
    return chunk_count


def process_file_paths(
    file_paths: list[str | Path],
    table_name: str | None = None,
    progress_callback: IngestionProgressCallback | None = None,
) -> tuple[bool, int, str | None]:
    tbl = table_name or DEFAULT_TABLE_NAME
    total_chunks = 0
    errors: list[str] = []

    def notify(
        file_path: str | Path,
        status: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(file_path, status, payload)

    try:
        for file_path in file_paths:
            try:
                print(f"Loading: {file_path}")
                notify(file_path, "parsing", None)
                docs = load_document_with_docling(file_path)
                if not docs:
                    message = "no document loaded"
                    errors.append(f"{Path(file_path).name}: {message}")
                    notify(file_path, "failed", {"message": message})
                    continue
                if not _has_extractable_text(docs):
                    message = "no text content extracted"
                    errors.append(f"{Path(file_path).name}: {message}")
                    notify(file_path, "failed", {"message": message})
                    continue
                notify(file_path, "embedding", None)
                num_chunks = _split_and_store(docs, table_name=tbl)
                if num_chunks == 0:
                    message = "no chunks inserted"
                    errors.append(f"{Path(file_path).name}: {message}")
                    notify(file_path, "failed", {"message": message})
                    continue
                total_chunks += num_chunks
                notify(file_path, "indexed", {"chunks_added": num_chunks})
            except Exception as exc:
                errors.append(f"{Path(file_path).name}: {exc}")
                notify(file_path, "failed", {"message": str(exc)})
                logger.error("process_file_paths file error %s: %s", file_path, exc)
            finally:
                _release_docling_converter()
        if total_chunks > 0:
            return True, total_chunks, None
        if errors:
            return False, 0, "; ".join(errors)
        return False, 0, "No documents loaded (unsupported type or empty)."
    except Exception as e:
        logger.error("process_file_paths error: %s", e)
        return False, 0, str(e)

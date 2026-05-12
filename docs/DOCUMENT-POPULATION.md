# Document Population Guide

This guide explains how to populate the vector database with documents using the shared ingestion implementation in `src/rag_agent/ingestion.py` and the supported CLI wrapper `scripts/ingest_documents.py`.

## Overview

The shared ingestion module in `src/rag_agent/ingestion.py` uses **Docling with RapidOCR** to convert uploaded files into Markdown, then delegates chunking and insertion to **OracleVS** (langchain-oracledb) with the app's `RecursiveCharacterTextSplitter`. The same table and embedding model are used at query time by the RAG app. The script entrypoint is now just a thin CLI wrapper over that module.

## What It Does

1. **Load**: Docling converts supported files into Markdown. PDF conversion has OCR enabled so scanned/image-only PDFs can be indexed when OCR recognizes text.
2. **File archival**: Copies each processed file to `uploaded_files/` and sets `source_url` as the primary source identity plus `file_name` for display in document metadata.
3. **Assign IDs**: Each loaded document gets a stable source/content ID when the loader did not provide one.
4. **Chunk and store**: OracleVS splits documents with RecursiveCharacterTextSplitter, derives chunk IDs from the document ID, embeds chunks using the app's embedding model (OCI), and inserts them into `RAG_KNOWLEDGE_BASE`.

## Supported File Formats

- **PDF** – Docling PDF conversion with RapidOCR enabled. Table-structure reconstruction is disabled for ingestion so scanned documents are converted as searchable text faster.
- **HTML / HTM** – Docling conversion.
- **TXT, MD, MARKDOWN** – Docling conversion.

## Prerequisites

1. Database connection configured in `.env` (VECTOR\_\* or CONNECT_ARGS)
2. OCI Generative AI credentials and embedding model set in `.env`
3. Dependencies installed (`uv sync`); requires `docling[rapidocr]`, `langchain-community`, `langchain-oracledb`, `langchain-oci`, and `langchain-text-splitters`.

## Usage

### Process specific files

```bash
# Single file
uv run python scripts/ingest_documents.py --files document.pdf

# Multiple files
uv run python scripts/ingest_documents.py --files doc1.pdf doc2.html notes.txt readme.md
```

### Process a directory

```bash
uv run python scripts/ingest_documents.py --dir ./documents
```

Loads all supported files under the given directory (recursive).

**Quick local example** (create a tiny doc and ingest it):

```bash
mkdir -p documents
printf "# Sample\nThis is a test document.\n" > documents/sample.md
uv run python scripts/ingest_documents.py --dir ./documents
```

### Command line arguments

- `--files` (optional): One or more file paths to process.
- `--dir` (optional): Directory path; all supported file types under it are loaded.
- Exactly one of `--files` or `--dir` is required.

## How It Works

1. **CLI wrapper**: `scripts/ingest_documents.py` parses command-line arguments and delegates to `src/rag_agent/ingestion.py`.
2. **Load**: For each supported file, Docling converts the content to Markdown; the file is copied to `uploaded_files/` and metadata (`source_url`, `file_name`, and related fields) is set on each `Document`.
3. **Prepare chunking**: The app creates RecursiveCharacterTextSplitter (chunk_size=800, chunk_overlap=150) and passes it to OracleVS.
4. **Store**: A DB connection is opened, the embedding model is obtained via `get_embedding_model()` (same as the RAG app), and `OracleVS.add_documents(..., text_splitter=...)` handles chunking, embedding, and insertion into `RAG_KNOWLEDGE_BASE` with COSINE distance.

## Output

- **Database**: Chunks in an OracleVS table with text, embeddings, and metadata (`source_url` first, plus `file_name`, `source_doc_index`, `chunk_index`, and related keys as available).
- **Files**: Processed files copied to `uploaded_files/` for citation links.
- **Console**: Progress messages and a final count of stored chunks.

## Configuration

Relevant settings in `.env` (see .env.example):

- `CONNECT_ARGS`: Database connection for OracleVS.
- `EMBED_MODEL_ID`, `COMPARTMENT_ID`, OCI settings: Used by `get_embedding_model()` from `oci_models`.

## Notes

- The shared ingestion module uses the same embedding model and OracleVS table shape as the RAG UI; citations and processed-source management work best when `metadata.source_url` is present.
- The first OCR run may download Docling/RapidOCR model artifacts and take longer than later runs. Set `HF_TOKEN` in the backend environment or pre-cache Docling model artifacts for more reliable production startup. If OCR extracts no text, ingestion returns an error rather than reporting a successful zero-chunk upload.

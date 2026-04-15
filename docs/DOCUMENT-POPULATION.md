# Document Population Guide

This guide explains how to populate the vector database with documents using the shared ingestion implementation in `src/rag_agent/ingestion.py` and the supported CLI wrapper `scripts/ingest_documents.py`.

## Overview

The shared ingestion module in `src/rag_agent/ingestion.py` uses **LangChain document loaders** (PyPDFLoader / UnstructuredFileLoader / TextLoader) to load files, **RecursiveCharacterTextSplitter** to chunk them, and **OracleVS** (langchain-oracledb) to embed and store chunks in the `RAG_KNOWLEDGE_BASE` table. The same table and embedding model are used at query time by the RAG app. The script entrypoint is now just a thin CLI wrapper over that module.

## What It Does

1. **Load**: LangChain loaders (PyPDFLoader, UnstructuredFileLoader, TextLoader) load files into LangChain `Document` objects.
2. **File archival**: Copies each processed file to `uploaded_files/` and sets `source_url` as the primary source identity plus `file_name` for display in document metadata.
3. **Chunk**: RecursiveCharacterTextSplitter splits documents into smaller chunks (default 800 characters, 150 overlap).
4. **Store**: OracleVS embeds chunks using the app’s embedding model (OCI) and inserts them into `RAG_KNOWLEDGE_BASE`.

## Supported File Formats

- **PDF** – PyPDFLoader (langchain_community)
- **HTML / HTM** – UnstructuredFileLoader
- **TXT, MD, MARKDOWN** – TextLoader

## Prerequisites

1. Database connection configured in `.env` (VECTOR\_\* or CONNECT_ARGS)
2. OCI Generative AI credentials and embedding model set in `.env`
3. Dependencies installed (`uv sync`); requires `langchain-community`, `langchain-oracledb`, `langchain-oci`, `langchain-text-splitters`

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
2. **Load**: For each file, the appropriate LangChain loader is used by extension; the file is copied to `uploaded_files/` and metadata (`source_url`, `file_name`, and related fields) is set on each `Document`.
3. **Split**: All documents are split with RecursiveCharacterTextSplitter (chunk_size=800, chunk_overlap=150).
4. **Store**: A DB connection is opened, the embedding model is obtained via `get_embedding_model()` (same as the RAG app), and `OracleVS.from_documents()` is called with the split documents, writing to `RAG_KNOWLEDGE_BASE` with COSINE distance.

## Output

- **Database**: Chunks in an OracleVS table with text, embeddings, and metadata (`source_url` first, plus `file_name` and related keys as available).
- **Files**: Processed files copied to `uploaded_files/` for citation links.
- **Console**: Progress messages and a final count of stored chunks.

## Configuration

Relevant settings in `.env` (see .env.example):

- `CONNECT_ARGS`: Database connection for OracleVS.
- `EMBED_MODEL_ID`, `COMPARTMENT_ID`, OCI settings: Used by `get_embedding_model()` from `oci_models`.

## Notes

- The shared ingestion module uses the same embedding model and OracleVS table shape as the RAG UI; citations and processed-source management work best when `metadata.source_url` is present.
- For PDF/HTML loaders, the `unstructured` library is used under the hood; ensure it is installed if you use those formats.

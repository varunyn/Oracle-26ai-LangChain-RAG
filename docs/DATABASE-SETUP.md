# Database setup requirements

## Overview

The RAG agent requires an **Oracle Vector Store** table with embeddings of your documents. At least one collection (table) must exist and contain vector data.

## Required database setup

### 1. Oracle Vector Store database

You need:

- **Oracle Database 23AI or 26AI** with Vector Store capabilities
- A database user with permissions to create/read tables
- Connection configured in `.env`:
  - `VECTOR_DB_USER`
  - `VECTOR_DB_PWD`
  - `VECTOR_DSN`
  - `VECTOR_WALLET_DIR` and `VECTOR_WALLET_PWD` (wallet-based connections)

See [CONFIGURATION.md](CONFIGURATION.md) for the full list.

### 2. Required table structure

Collections must match the schema used by `langchain-oracledb` (OracleVS). These columns are required by the current LangChain retrieval and ingestion path:

- `id` (RAW(16) primary key)
- `text` (CLOB)
- `metadata` (JSON)
- `embedding` (VECTOR(dim, FLOAT32))

Example (DDL):

```sql
CREATE TABLE RAG_KNOWLEDGE_BASE (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    text CLOB,
    metadata JSON,
    embedding VECTOR(1536, FLOAT32)
);
```

> The embedding dimension **must** match your embedding model. The app expects an OracleVS-compatible table shape even when the collection name follows an `ORACLE_WEB_EMBEDDINGS` naming convention.

### 3. Create the table (recommended)

This repo ships a helper script that builds an OracleVS-compatible table using your config:

```bash
uv run scripts/create_rag_table.py --table RAG_KNOWLEDGE_BASE --yes
```

If you need a custom embedding dimension, pass `--embedding-dim` (see script header for examples).

### 4. Collections used by the app

By default the app uses:

- `COLLECTION_LIST=RAG_KNOWLEDGE_BASE`
- `DEFAULT_COLLECTION=RAG_KNOWLEDGE_BASE`

Update these in `.env` if your table name differs.

### 5. Metadata convention used by the app

Beyond the required OracleVS columns, this app expects source metadata inside the `metadata` JSON document. Going forward, the preferred convention is:

- `source_url` — **primary source identity** for listing, deletion, and source management UI
- `file_name` — short label for display when available
- `source` — optional display field
- `chunk_offset` and/or page metadata — helps citations show chunk/page location

If you populate tables outside this app, keep the OracleVS columns above and make sure `metadata.source_url` is present.

### 6. Load data into the collection

Use the supported ingestion CLI wrapper to ingest files. The implementation lives in `src/rag_agent/ingestion.py`, and `scripts/ingest_documents.py` is the operator-facing entrypoint:

```bash
uv run python scripts/ingest_documents.py --files "document.pdf" "notes.txt" "readme.md"
```

See [DOCUMENT-POPULATION.md](DOCUMENT-POPULATION.md) for details.

### 7. What the app expects

The app will:

1. **List collections** using tables with `VECTOR` columns
2. **Search** in those collections with vector similarity
3. **Read metadata** from the JSON column for citations and source labels

## Troubleshooting

- **No collections found**: verify your vector table exists and the user can read it.
- **Empty search results**: ingest documents into the collection.
- **Connection errors**: confirm wallet path and `VECTOR_*` values in `.env`.

## Note

You can use any table name; just set `COLLECTION_LIST` and `DEFAULT_COLLECTION` in `.env` to match.

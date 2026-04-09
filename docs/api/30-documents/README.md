# Documents

## POST `/api/documents/upload`

Upload supported files and ingest them into the configured OracleVS-compatible vector collection.

### Multipart form fields

- `files`: one or more uploaded files
- `collection_name`: optional form field

### Supported file types

- pdf
- html
- htm
- txt
- md
- markdown

### Example curl

```bash
curl -X POST http://127.0.0.1:3002/api/documents/upload \
  -F "files=@./example.pdf" \
  -F "collection_name=RAG_KNOWLEDGE_BASE"
```

### Success response

```json
{
  "chunks_added": 42,
  "files_processed": 1,
  "collection": "RAG_KNOWLEDGE_BASE"
}
```

### Metadata notes

Uploaded documents are stored in an OracleVS-compatible table and the ingestion path writes `metadata.source_url` as the preferred source identity for processed-source management, along with display-oriented metadata such as `file_name`.

### Failure modes

Common error responses include:

- no files provided
- no supported files
- document upload not available
- processing failed

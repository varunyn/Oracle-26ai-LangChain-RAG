# API Docs

This directory contains human-friendly product API documentation plus Bruno request artifacts for the FastAPI routes mounted into the LangGraph Agent Server.

## Source of truth

The API contract is primarily defined by FastAPI request/response models and generated OpenAPI:

- Runtime OpenAPI: `app.openapi()`
- Export script: `uv run python scripts/export_openapi.py tests/fixtures/openapi-baseline.json`
- Regression check: `uv run pytest tests/workflow_tests/test_openapi_baseline.py -q`

Use the generated OpenAPI for schema truth and these docs for workflow guidance, streaming behavior, and ready-to-run examples.

## Structure

- `00-overview/` — base URL, headers, auth assumptions, streaming contract
- `10-health/` — health checks
- `20-chat/` — thread/run chat endpoints
- `30-documents/` — document upload
- `60-config-suggestions-feedback/` — config, suggestions, feedback
- `environments/` — sample environment values
- `bruno/` — Bruno collection and requests

## FastAPI automation

FastAPI already generates OpenAPI automatically from:

- route decorators
- Pydantic models
- response models
- parameter types and metadata
- docstrings

That means most schema docs can be generated automatically. These markdown files exist to document what OpenAPI alone does not explain well, especially:

- SSE stream behavior
- required headers
- request sequencing
- practical examples for local development
- Bruno workflows

## Local base URL

Backend default:

```text
http://127.0.0.1:2024
```

## Quick checks

```bash
curl -s http://127.0.0.1:2024/health
uv run pytest tests/workflow_tests/test_openapi_baseline.py -q
uv run pytest tests/workflow_tests/test_api_docs_sync.py -q
uv run python scripts/sync_api_docs.py --check
./scripts/streaming_smoke_test.sh
```

## Bruno

A starter Bruno collection lives under `docs/api/bruno/CustomRAGAgent`.

Recommended workflow:

1. Start the Agent Server with `uv run langgraph dev`
2. Open the Bruno collection directory
3. Select the `local` environment
4. Run health first, then JSON endpoints, then streaming endpoints

## Important contract notes

### Chat protocol ownership

FastAPI no longer owns chat thread/run/stream endpoints.

Frontend chat now targets LangGraph Agent Server directly through
`NEXT_PUBLIC_LANGGRAPH_API_BASE` and `assistantId: "chat_agent"`.

See `20-chat/README.md` for details.

### MCP-enabled chat

MCP-enabled chat is supported through the Agent Server `chat_agent` graph using
`mode="mcp"` or `mode="mixed"` in top-level run `context`.

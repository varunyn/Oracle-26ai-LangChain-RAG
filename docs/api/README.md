# API Docs

This directory contains human-friendly API documentation plus Bruno request artifacts for the FastAPI backend.

## Source of truth

The API contract is primarily defined by FastAPI request/response models and generated OpenAPI:

- Runtime OpenAPI: `app.openapi()`
- Export script: `uv run python scripts/export_openapi.py tests/fixtures/openapi-baseline.json`
- Regression check: `uv run pytest tests/test_openapi_baseline.py -q`

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
http://127.0.0.1:3002
```

## Quick checks

```bash
curl -s http://127.0.0.1:3002/health
uv run pytest tests/workflow_tests/test_openapi_baseline.py -q
uv run pytest tests/workflow_tests/test_api_docs_sync.py -q
uv run python scripts/sync_api_docs.py --check
./scripts/streaming_smoke_test.sh
```

## Bruno

A starter Bruno collection lives under `docs/api/bruno/CustomRAGAgent`.

Recommended workflow:

1. Start the API with `./run_api.sh`
2. Open the Bruno collection directory
3. Select the `local` environment
4. Run health first, then JSON endpoints, then streaming endpoints

## Important contract notes

### `/api/langgraph/threads/{thread_id}/runs`

Thread/run endpoints are the primary contract for frontend chat:

- `messages` must contain **at least one** user message
- the **final** message must have `role="user"`
- earlier system/user/assistant messages are treated as chat history
- use `/runs/stream` for SSE `event: values` frames
- stream completion is transport close (no `[DONE]`)

See `20-chat/README.md` for details.

### MCP-enabled chat

MCP-enabled chat is supported through thread/run endpoints using `mode="mcp"` or `mode="mixed"`.

# API Overview

## Base URL

```text
http://127.0.0.1:3002
```

## Content types

Most endpoints use JSON.

Streaming endpoints use:

```text
Content-Type: text/event-stream
```

## Request ID

The API uses request ID middleware and returns `X-Request-ID` for correlation. When debugging, preserve and log this header.

## Auth

There is no general bearer-token auth layer documented for these endpoints today. Any deployment-specific auth or proxy auth should be documented separately from this API reference.

## Generated docs vs curated docs

Use FastAPI-generated OpenAPI for schemas.
Use this docs tree for:

- endpoint grouping
- examples
- SSE behavior
- Bruno usage
- environment setup

## Streaming contract summary

For `/api/langgraph/threads/{thread_id}/runs/stream` responses:

- content-type: `text/event-stream`
- event name: `values`
- frame payload: `{"messages":[...]}`
- completion: stream close (no `[DONE]` frame)

These are contract requirements between frontend and backend.

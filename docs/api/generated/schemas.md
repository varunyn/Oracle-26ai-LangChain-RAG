# Generated Schema Reference

This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.
Do not edit manually.

## `Body_upload_documents_api_documents_upload_post`

- type: `object`

### Properties

- `collection_name`: `complex`
- `files`: `array`

## `FeedbackRequest`

- type: `object`
- required: `question`, `answer`, `feedback`

### Properties

- `answer`: `string` Assistant answer
- `feedback`: `integer` Star rating 1-5
- `question`: `string` User question

## `HTTPValidationError`

- type: `object`

### Properties

- `detail`: `array`

## `RunInput`

- type: `object`

### Properties

- `collection_name`: `complex`
- `enable_reranker`: `complex`
- `enable_tracing`: `complex`
- `mcp_server_keys`: `complex`
- `message`: `complex`
- `messages`: `complex`
- `mode`: `complex`
- `model`: `complex`
- `session_id`: `complex`

## `SuggestionsRequest`

- type: `object`
- required: `last_message`

### Properties

- `last_message`: `string` Last assistant message text to base suggestions on
- `last_user_message`: `complex` Latest user question to keep suggestions on-topic
- `model`: `complex` Model ID; uses default if omitted

## `SuggestionsResponse`

- type: `object`

### Properties

- `suggestions`: `array` Follow-up question strings

## `ThreadCreateRequest`

- type: `object`

### Properties

- `thread_id`: `complex`

## `ThreadCreateResponse`

- type: `object`
- required: `thread_id`

### Properties

- `thread_id`: `string`

## `ThreadHistoryRequest`

- type: `object`

### Properties

- `before`: `complex`
- `checkpoint`: `complex`
- `limit`: `complex`
- `metadata`: `complex`

## `ThreadRunRequest`

- type: `object`

### Properties

- `assistant_id`: `complex`
- `collection_name`: `complex`
- `configurable`: `complex`
- `context`: `complex`
- `enable_reranker`: `complex`
- `enable_tracing`: `complex`
- `input`: `complex`
- `mcp_server_keys`: `complex`
- `message`: `complex`
- `messages`: `complex`
- `metadata`: `complex`
- `mode`: `complex`
- `model`: `complex`
- `session_id`: `complex`

## `ThreadRunResponse`

- type: `object`
- required: `run_id`, `thread_id`, `output`

### Properties

- `output`: `object`
- `run_id`: `string`
- `thread_id`: `string`

## `ValidationError`

- type: `object`
- required: `loc`, `msg`, `type`

### Properties

- `ctx`: `object`
- `input`: `complex`
- `loc`: `array`
- `msg`: `string`
- `type`: `string`

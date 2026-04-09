# Generated Schema Reference

This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.
Do not edit manually.

## `Body_upload_documents_api_documents_upload_post`

- type: `object`

### Properties

- `collection_name`: `complex`
- `files`: `array`

## `ChatCompletionsRequest`

- type: `object`
- required: `messages`

### Properties

- `collection_name`: `complex` Vector store collection/table name
- `enable_reranker`: `complex` Enable reranker step
- `enable_tracing`: `complex` Enable tracing
- `mcp_server_keys`: `complex` MCP server keys from MCP_SERVERS_CONFIG to load tools
- `messages`: `array` Conversation messages (OpenAI-compatible format)
- `mode`: `complex` Flow mode: rag | mcp | mixed | direct; default rag for backward compat
- `model`: `complex` Model ID
- `session_id`: `complex` Browser/session ID for grouping traces (new per tab load or refresh)
- `stream`: `boolean` If true, return SSE stream
- `thread_id`: `complex` Conversation thread ID for checkpointer memory

## `ChatMessage`

- type: `object`
- required: `role`, `content`

### Properties

- `content`: `string`
- `role`: `string`

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

## `InvokeRequest`

- type: `object`
- required: `user_input`

### Properties

- `user_input`: `string`

## `McpChatRequest`

- type: `object`
- required: `messages`

### Properties

- `mcp_url`: `complex` MCP server URL (e.g. http://host:port/mcp/); uses default if not set)
- `messages`: `array` Conversation messages (OpenAI format)
- `stream`: `boolean` If true, return SSE stream

## `SuggestionsRequest`

- type: `object`
- required: `last_message`

### Properties

- `last_message`: `string` Last assistant message text to base suggestions on
- `model`: `complex` Model ID; uses default if omitted

## `SuggestionsResponse`

- type: `object`

### Properties

- `suggestions`: `array` Follow-up question strings

## `ValidationError`

- type: `object`
- required: `loc`, `msg`, `type`

### Properties

- `ctx`: `object`
- `input`: `complex`
- `loc`: `array`
- `msg`: `string`
- `type`: `string`

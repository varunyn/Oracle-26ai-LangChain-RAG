# Generated Schema Reference

This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.
Do not edit manually.

## `AppConfigResponse`

- type: `object`
- required: `region`, `embed_model_id`, `model_list`, `model_display_names`, `collection_list`, `enable_user_feedback`, `observability`

### Properties

- `collection_list`: `array`
- `embed_model_id`: `string`
- `enable_user_feedback`: `boolean`
- `model_display_names`: `object`
- `model_list`: `array`
- `observability`: `complex`
- `region`: `string`

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
- `trace_id`: `complex` Langfuse trace id for this answer

## `HTTPValidationError`

- type: `object`

### Properties

- `detail`: `array`

## `MCPConnectionTestResponse`

- type: `object`
- required: `key`, `ok`, `tool_count`, `tools`, `error`

### Properties

- `error`: `complex`
- `key`: `string`
- `ok`: `boolean`
- `tool_count`: `integer`
- `tools`: `array`

## `MCPConnectionToolResponse`

- type: `object`
- required: `name`, `description`

### Properties

- `description`: `string`
- `name`: `string`

## `MCPServerAuthResponse`

- type: `object`

### Properties

- `audience`: `complex`
- `bearer_token_set`: `boolean`
- `client_id`: `complex`
- `client_secret_set`: `boolean`
- `grant_type`: `string`
- `refresh_skew_seconds`: `integer`
- `scope`: `complex`
- `token_url`: `complex`
- `type`: `string`

## `MCPServerAuthWrite`

- type: `object`

### Properties

- `audience`: `complex`
- `bearer_token`: `complex`
- `client_id`: `complex`
- `client_secret`: `complex`
- `grant_type`: `string`
- `refresh_skew_seconds`: `integer`
- `scope`: `complex`
- `token_url`: `complex`
- `type`: `string`

## `MCPServerConfigResponse`

- type: `object`
- required: `key`, `transport`, `url`, `enabled`, `auth`

### Properties

- `auth`: `complex`
- `enabled`: `boolean`
- `key`: `string`
- `transport`: `string`
- `url`: `string`

## `MCPServerConfigWrite`

- type: `object`
- required: `url`

### Properties

- `auth`: `complex`
- `enabled`: `boolean`
- `transport`: `string`
- `url`: `string`

## `MCPServerEnabledWrite`

- type: `object`
- required: `enabled`

### Properties

- `enabled`: `boolean`

## `MCPServersConfigResponse`

- type: `object`
- required: `enable_mcp_tools`, `servers`

### Properties

- `enable_mcp_tools`: `boolean`
- `servers`: `array`

## `ObservabilityConfigResponse`

- type: `object`
- required: `links`

### Properties

- `links`: `array`

## `ObservabilityLinkResponse`

- type: `object`
- required: `key`, `label`, `enabled`, `configured`, `status`, `details`

### Properties

- `configured`: `boolean`
- `details`: `string`
- `enabled`: `boolean`
- `key`: `string`
- `label`: `string`
- `status`: `string`
- `url`: `complex`

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
- `stream_mode`: `complex`

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
- `stream_mode`: `complex`

## `ValidationError`

- type: `object`
- required: `loc`, `msg`, `type`

### Properties

- `ctx`: `object`
- `input`: `complex`
- `loc`: `array`
- `msg`: `string`
- `type`: `string`

# Chat API

This app exposes two main chat-style entrypoints:

- `POST /invoke`
- `POST /api/chat`

`/api/chat` is the primary public API and the supported entrypoint for RAG-only, MCP-only, and mixed RAG+MCP chat.

## Request model highlights

`ChatCompletionsRequest` fields include:

- `model`
- `messages`
- `stream`
- `thread_id`
- `session_id`
- `collection_name`
- `enable_reranker`
- `enable_tracing`
- `mode`
- `mcp_server_keys`

`mcp_server_keys` limits which configured MCP servers/tools are loaded when MCP is enabled. It does not by itself choose the chat mode; use `mode="mcp"` or `mode="mixed"` for MCP-enabled chat.

## Important validation rule

`messages` must contain at least one `user` message, and the final message must have `role: "user"`.

`/api/chat` accepts full message history in OpenAI-style order. The backend uses earlier
messages as chat history and treats the final user message as the current request.

## POST `/api/chat`

### Non-stream example

```json
{
  "model": "cohere.command-r-plus",
  "messages": [
    {
      "role": "user",
      "content": "Answer with a markdown table when appropriate."
    },
    {
      "role": "assistant",
      "content": "Understood."
    },
    {
      "role": "user",
      "content": "What documents mention Oracle vector search?"
    }
  ],
  "stream": false,
  "thread_id": "demo-thread",
  "collection_name": "RAG_KNOWLEDGE_BASE"
}
```

### Non-stream response shape

Returns a JSON object including:

- `content`
- `choices`
- `usage`
- `standalone_question`
- `citations`
- `reranker_docs`
- `context_usage`

### Stream example

Use the same request body but set `stream: true`.

### Streaming behavior

When `stream=true`, the response is SSE and includes these important headers:

- `content-type: text/event-stream`
- `x-vercel-ai-ui-message-stream: v1`
- `cache-control: no-cache`
- `x-accel-buffering: no`

The stream ends with:

```text
data: [DONE]
```

### Recommended verification

```bash
./scripts/streaming_smoke_test.sh
```

## POST `/invoke`

This endpoint is a simpler invoke-style entrypoint and is mainly useful for backend-level invocation/testing.

Prefer `/api/chat` for documented client integrations.

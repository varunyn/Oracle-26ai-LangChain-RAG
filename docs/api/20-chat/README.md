# Chat API

This app exposes thread/run chat endpoints:

- `POST /api/langgraph/threads`
- `POST /api/langgraph/threads/{thread_id}/runs`
- `POST /api/langgraph/threads/{thread_id}/runs/stream`

## Request model highlights

`ThreadRunRequest` accepts:

- top-level `messages` or nested `input.messages`
- top-level `message` or nested `input.message`
- optional runtime options (`model`, `session_id`, `collection_name`, `enable_reranker`, `enable_tracing`, `mode`, `mcp_server_keys`) at top level or under `input`

`mcp_server_keys` limits which configured MCP servers/tools are loaded when MCP is enabled. It does not by itself choose the chat mode; use `mode="mcp"` or `mode="mixed"` for MCP-enabled chat.

## Important validation rule

Thread/run payloads must provide either `input.messages` (with at least one user/human message) or
`input.message`.

## POST `/api/langgraph/threads/{thread_id}/runs`

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

Use `POST /api/langgraph/threads/{thread_id}/runs/stream` with the same request body.

### Streaming behavior

The response is SSE (`text/event-stream`) using repeated `event: values` frames.

Each frame contains a full `messages` snapshot, for example:

```text
event: values
data: {"messages":[...]}
```

There is no `[DONE]` sentinel; completion is stream close.

### Recommended verification

```bash
./scripts/streaming_smoke_test.sh
```

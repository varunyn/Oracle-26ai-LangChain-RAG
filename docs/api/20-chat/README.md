# Chat API

This app exposes thread/run chat endpoints:

- `POST /api/langgraph/threads`
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

## POST `/api/langgraph/threads/{thread_id}/runs/stream`

### Stream request example

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

### Streaming behavior

The response is SSE (`text/event-stream`) using repeated `event: values` frames.

Each frame contains a full `messages` snapshot, for example:

```text
event: values
data: {"messages":[...]}
```

There is no `[DONE]` sentinel; completion is stream close.

Assistant message metadata carries references, sources, tool activity, and MCP invocation details. The frontend consumes this values-stream through `@langchain/react` and renders answer text, tool activity, and sources from the latest assistant snapshot.

For `mode="mixed"`, the runtime loads MCP tools plus the local `oracle_retrieval` tool. If a turn only needs retrieval and retrieval returns documents, the MCP agent turn stops after retrieval and the final answer streams through the RAG answer path. If the user explicitly asks for another MCP tool/action, the agent can continue after retrieval and that tool result is included as supplemental context for the RAG answer. If retrieval is not used or no documents are returned, the response is the MCP agent answer or a retrieval error.

### Recommended verification

```bash
./scripts/streaming_smoke_test.sh
```

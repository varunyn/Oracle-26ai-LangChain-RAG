# LangChain-Native Chat Memory Refactor

## Goal

Remove the project-specific dictionary conversion and legacy message compatibility layer from the active chat graph runtime. LangGraph's `add_messages` reducer already deserializes supported graph inputs into LangChain messages, so the graph should keep `AnyMessage` objects internally.

## Design

- `ChatGraphState.messages` remains `list[AnyMessage]` and is the canonical internal representation.
- Direct, RAG, MCP, and mixed nodes pass state messages directly to LLM/history helpers.
- `latest_user_message` and `chat_history_before_latest_user` accept LangChain message sequences.
- Delete `to_langchain_messages`, `langchain_messages_to_dicts`, stringified-content repair, role-alias normalization, and unused merge/deduplication helpers.
- Preserve normal LangChain structured content and message IDs; do not transform or reconstruct messages in the runtime.
- Replace unit tests that assert legacy dictionary behavior with tests for native LangChain messages, structured content, and IDs.

## Compatibility boundary

The graph input boundary remains LangGraph's built-in message deserialization. API/frontend contracts outside `src/rag_agent/runtime/memory.py` are not changed by this refactor.

## Verification

- Focused memory unit tests.
- Relevant graph/workflow tests for direct, RAG, MCP, and mixed execution.
- Ruff for changed Python files.
- Review the final diff to ensure no converter or legacy helper references remain.

# Changelog

## 2026-06-25

- Migrated LangGraph chat streaming to the `@langchain/react` v1 command and protocol event endpoints.
- Updated LangChain/LangGraph-related runtime dependencies and removed the local `langchain-oracle` Docker build override.
- Refreshed API docs, Bruno requests, and OpenAPI fixtures for `/commands` and `/stream/events`.
- Fixed frontend chat rendering so submitted suggestion prompts are not rendered twice.

# Changelog

## 2026-06-25

- Migrated LangGraph chat streaming to the `@langchain/react` v1 command and protocol event endpoints.
- Updated LangChain/LangGraph-related runtime dependencies and removed the local `langchain-oracle` Docker build override.
- Refreshed API docs, Bruno requests, and OpenAPI fixtures for `/commands` and `/stream/events`.
- Fixed frontend chat rendering so submitted suggestion prompts are not rendered twice.

## 2026-06-26

- Split Playwright coverage into deterministic mocked chat-streaming tests and live-backend RAG chat tests.
- Switched the Playwright frontend web server default port to `4040` to keep e2e runs separate from the normal frontend dev port.
- Aligned LangGraph protocol user-message ids between frontend optimistic values and backend current-turn values so submitted questions render once and before the assistant response.
- Added the initial LangGraph Agent Server bootstrap surface with `langgraph.json` and a minimal `chat_agent` graph while the legacy chat surface still coexists during the compatibility phase.
- Fixed LangGraph direct/RAG routing to use runtime context instead of graph state, reject unsupported `mcp`/`mixed` modes explicitly, move blocking RAG retrieval work off the Agent Server event loop, and cover the graph-mode contract with deterministic workflow tests plus real SDK integration tests.
- Added LangGraph graph-owned `mcp` and `mixed` routes, including live MCP/calculator and mixed retrieval-plus-tool verification against the Agent Server.
- Fixed LangGraph MCP config resolution to load the UI-managed MCP server store from the repo root during Agent Server runs.

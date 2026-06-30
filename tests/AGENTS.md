## Testing Instructions For LangChain Projects

The graph/server contract is the active chat surface. Workflow tests should exercise graph construction, Agent Server bootstrap/context, message shape, and mode routing; frontend tests cover projection/replay/rendering separately.

When writing or modifying tests in this repository, follow these rules for all LangChain-based agents, chains, tools, and integrations.

Use the LangChain docs MCP and check the latest unit-testing and integration-testing docs before changing testing patterns.

### 1. Maintain a strict split between test categories

- Put **unit tests** in `tests/unit_tests`.
- Put **workflow tests** in `tests/workflow_tests`.
- Put **integration tests** in `tests/integration_tests`.
- Keep manual scripts as `tests/run_*.py`; they are not pytest suites.

Use these meanings consistently:

- `tests/unit_tests`: fast, deterministic, no real network, no real provider/backend calls.
- `tests/workflow_tests`: deterministic multi-node or orchestration tests with mocked/fake external boundaries.
- `tests/integration_tests`: real external/provider/backend/runtime tests only.

Do not mix real API calls into unit or workflow tests. Do not label mocked orchestration tests as integration tests.

### 2. Unit tests must not call real LLMs or external services

For unit tests:

- Prefer `GenericFakeChatModel` for scripted chat-model behavior.
- Script exact responses using plain strings or `AIMessage` objects.
- Keep tests deterministic and repeatable.
- Never require API keys, OCI credentials, MCP server availability, or live DB access.

LangChain recommends fake chat models so unit tests stay fast, free, and reproducible.

### 3. Use the shared helpers before inventing new fake classes

Shared helpers live in `tests/unit_tests/helpers.py`.

Prefer these first:

- `fake_chat_model(...)` for plain deterministic chat responses
- `ToolBindableFakeChatModel` when production code expects `bind_tools()`
- `StructuredOutputFakeChatModel` when production code expects `with_structured_output()`
- `tool_call_message(...)` for repeated `AIMessage(tool_calls=...)` fixtures

Important: do not assume `GenericFakeChatModel` actually implements advanced inherited helpers just because the method exists on a base class. In this repo, direct `with_structured_output()` on `GenericFakeChatModel` has been verified to raise `NotImplementedError`. When fake models do not support a needed capability, prefer the shared adapter helpers instead of introducing new inline fake classes.

### 4. Prefer fake model fixtures for agent logic

When testing agent logic, script sequences that may include:

- plain text responses
- `AIMessage` outputs
- tool calls
- failures or edge cases

Use this to verify control flow, tool selection, message handling, and error paths without depending on provider behavior.

### 5. Test thread-state behavior with runtime fixtures

If a compatibility/runtime unit test depends on memory or thread state:

- use the current compatibility runtime's in-memory thread state in tests
- simulate multiple turns with the same `thread_id`
- verify state-dependent behavior without external persistence

The browser chat uses LangGraph Agent Server persistence; do not add LangGraph-specific checkpointer setup to compatibility unit tests unless the test explicitly targets the Agent Server graph.

### 6. Integration tests must use real APIs intentionally

Integration tests should validate that:

- provider credentials are valid
- the selected model can actually serve requests
- tools and external services work together correctly
- the end-to-end wiring behaves as expected
- latency and runtime behavior are acceptable

Integration tests are allowed to make real network calls, but only in `tests/integration_tests` and they should be explicitly marked, usually with `@pytest.mark.integration`.

For integration assertions:

- prefer structure and contract assertions over exact freeform wording
- use VCR where HTTP replay is appropriate
- keep real-boundary scope narrow when possible

### 7. Keep citation assertions aligned with the shared normalizer

- Citation shaping is centralized in `src/rag_agent/core/citations.py`.
- When asserting citations in unit/workflow tests, validate the normalized contract (`source`, `page`, `link`) and preserve optional passthrough fields when present (for example `score`).
- Prefer updating tests to reflect shared normalizer behavior rather than reintroducing per-route or per-service citation mapping logic.

### 8. Streaming and thread-state assertions

- Assert structured assistant content and stable message ids; do not assert Python-list stringification.
- When testing replay behavior, use the same thread id across turns and verify persisted messages are not rendered twice.
- For live Agent Server coverage, use the tests in `tests/integration_tests/` and explicitly gate credentials, database, MCP, and model requirements.

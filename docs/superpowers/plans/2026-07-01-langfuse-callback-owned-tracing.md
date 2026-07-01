# Langfuse Callback-Owned Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make mixed-mode LangGraph traces follow the documented Langfuse LangChain integration with one callback-owned request trace and automatic nested LLM/tool/retrieval observations.

**Architecture:** Expose a LangGraph graph factory that receives the Agent Server `RunnableConfig`, attaches one standard Langfuse `CallbackHandler` at the graph boundary, and returns the compiled graph with that configuration bound. Remove node-level manual root traces and per-node callback creation; inner model, agent, tool, and retrieval calls inherit the graph callback through their existing `RunnableConfig` propagation.

**Tech Stack:** Python 3.11+, LangGraph Agent Server, LangChain `RunnableConfig`, Langfuse Python SDK 4.9.1, pytest, Ruff.

## Global Constraints

- Use the documented Langfuse LangChain callback integration; do not add custom child observations or finish-reason classifiers.
- Keep the graph's public `chat_agent` behavior, mode routing, message contracts, and persistence unchanged.
- Preserve complete local Langfuse inputs and outputs; no masking or truncation options.
- Do not add fallback tracing paths. If tracing is disabled or not configured, the graph runs without a callback.
- Keep secrets in environment/settings; never place Langfuse credentials in code or tests.
- Record the significant tracing change in `CHANGELOG.md` under `2026-07-01`.

---

### Task 1: Add graph-boundary callback coverage

**Files:**
- Modify: `src/rag_agent/graphs/chat_agent.py`
- Test: `tests/workflow_tests/test_langgraph_server_bootstrap.py`
- Test: `tests/workflow_tests/test_langgraph_chat_agent_modes.py`

**Interfaces:**
- Produce `make_chat_agent(config: RunnableConfig | None = None)`, a graph factory compatible with the `langgraph.json` graph entrypoint.
- Keep `build_chat_agent(checkpointer=...)` available for unit/workflow tests and local graph construction.

- [x] **Step 1: Write the failing factory test**

  Assert that the factory receives a config with `configurable.enable_tracing=True`, attaches the Langfuse callback configuration once, and returns a runnable graph. The test must monkeypatch the callback attachment helper and capture the supplied config; it must also assert tracing is not attached when `enable_tracing=False`.

- [x] **Step 2: Run the focused test and verify it fails**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_server_bootstrap.py -q`

  Expected: FAIL because `make_chat_agent` is not yet exported and `langgraph.json` still points to the static graph variable.

- [x] **Step 3: Implement the minimal factory**

  Add a factory that copies the incoming `RunnableConfig`, calls the existing callback attachment helper once when tracing is enabled, and returns `build_chat_agent().with_config(config)`. Do not create observations or callbacks inside graph nodes.

- [x] **Step 4: Run the focused test and verify it passes**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_server_bootstrap.py tests/workflow_tests/test_langgraph_chat_agent_modes.py -q`

  Expected: PASS before removing node-level tracing.

- [x] **Step 5: Review checkpoint**

  Commit subject: `refactor: bind Langfuse tracing at graph boundary`

### Task 2: Switch Agent Server to the graph factory

**Files:**
- Modify: `langgraph.json`
- Modify: `tests/workflow_tests/test_langgraph_server_bootstrap.py`

**Interfaces:**
- Agent Server resolves `chat_agent` through `src.rag_agent.graphs.chat_agent:make_chat_agent`.

- [x] **Step 1: Update the bootstrap expectation**

  Change the expected graph entrypoint from the static `chat_agent` variable to `make_chat_agent`.

- [x] **Step 2: Run the bootstrap test and verify it fails before implementation**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_server_bootstrap.py -q`

- [x] **Step 3: Update `langgraph.json`**

  Point the `chat_agent` graph to the factory function.

- [x] **Step 4: Run graph bootstrap and contract tests**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_server_bootstrap.py tests/workflow_tests/test_langgraph_chat_contract.py -q`

### Task 3: Remove node-level manual roots and duplicate callback injection

**Files:**
- Modify: `src/rag_agent/graphs/nodes/direct.py`
- Modify: `src/rag_agent/graphs/nodes/rag.py`
- Modify: `src/rag_agent/graphs/nodes/mcp.py`
- Modify: `src/rag_agent/graphs/nodes/mixed.py`
- Modify: `src/rag_agent/graphs/runtime.py`
- Modify: `src/rag_agent/utils/langfuse_tracing.py`
- Modify: `tests/unit_tests/test_langfuse_tracing.py`
- Modify: `tests/workflow_tests/test_langgraph_chat_agent_modes.py`

**Interfaces:**
- Existing mode nodes continue passing the inherited `RunnableConfig` to every inner LangChain call.
- `build_run_config` preserves metadata/configurable values but no longer adds another Langfuse callback.
- Mode result contracts no longer manufacture a node-local `trace_id`; Agent Server/Langfuse owns trace identity.

- [x] **Step 1: Update tests to assert inherited callbacks are not duplicated**

  Add a test that supplies a parent config containing one callback and verifies `build_run_config` preserves it without adding a second Langfuse handler.

- [x] **Step 2: Run the focused test and verify it fails**

  Run: `./.venv/bin/pytest tests/unit_tests/test_langfuse_tracing.py tests/workflow_tests/test_langgraph_chat_agent_modes.py -q`

- [x] **Step 3: Remove manual root context blocks**

  In each mode node, use the node's `config` where available and pass the inherited config to inner calls. Remove `start_langfuse_chat_trace`, `trace_context`, manual trace output updates, and node-local trace IDs. In direct/RAG nodes, preserve the existing result data and usage observability.

- [x] **Step 4: Remove callback injection from `build_run_config`**

  Keep config metadata construction. Delete the `add_langfuse_callbacks` call and tracing-only imports from this helper so the graph-bound callback is the sole callback owner.

- [x] **Step 5: Confirm root API remains only for standalone suggestions**

  Delete `start_langfuse_chat_trace` and its tests if no product/API caller remains. Retain client setup, callback creation, flush/shutdown, and usage handling that are still used by the graph-bound handler or product suggestions route.

- [x] **Step 6: Run mode workflow tests**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_chat_agent_modes.py tests/workflow_tests/test_langgraph_chat_contract.py tests/unit_tests/test_langfuse_tracing.py -q`

### Task 4: Validate the real mixed-mode trace contract

**Files:**
- Modify: `CHANGELOG.md`
- Potentially modify: `docs/CONFIGURATION.md` only if the final supported tracing configuration changes.

- [x] **Step 1: Run static checks**

  Run: `./.venv/bin/ruff check src/rag_agent/graphs src/rag_agent/utils/langfuse_tracing.py tests`

- [x] **Step 2: Run the relevant workflow suite**

  Run: `./.venv/bin/pytest tests/workflow_tests/test_langgraph_chat_agent_modes.py tests/workflow_tests/test_langgraph_chat_contract.py tests/workflow_tests/test_langgraph_server_bootstrap.py tests/workflow_tests/test_chat_nonstream_and_validation.py -q`

- [ ] **Step 3: Exercise one real mixed-mode Agent Server request**

  With the local LangGraph/Langfuse services running and tracing enabled, send one mixed-mode request and inspect the trace. Verify one request trace contains the model generations, MCP tool observations, retrieval work where applicable, final output, session/request/model metadata, and usage/cost data.

- [x] **Step 4: Record the completed architecture change**

  Add a `2026-07-01` changelog entry explaining that Langfuse tracing is now attached once at the Agent Server graph boundary through the standard callback handler.

- [x] **Step 5: Run final diff and regression checks**

  Run: `git diff --check` and `./scripts/regression_guard.sh` when the local services required by the guard are available.

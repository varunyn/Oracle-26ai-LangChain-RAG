# LangGraph Native Chat Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move chat execution from a thin LangGraph wrapper around `ChatRuntimeService` to graph-owned mode nodes while keeping FastAPI as the supported custom HTTP route layer.

**Architecture:** LangGraph Agent Server remains the canonical chat/thread/run/streaming surface. FastAPI stays as `langgraph.json` `http.app` for product APIs such as config, MCP settings, suggestions, feedback, documents, and health. Runtime behavior is split into graph nodes and small domain services so LangGraph state, context, checkpoints, and streaming carry the chat lifecycle instead of a monolithic service doing the whole turn.

**Tech Stack:** Python 3.11, LangGraph 1.x, LangChain messages/runnables, FastAPI custom routes, `@langchain/react`, pytest, Docker Compose.

## Global Constraints

- Do not remove FastAPI. Current LangGraph docs support mounting a Python FastAPI app via `langgraph.json` `http.app`.
- Do not restore `/api/langgraph/*` compatibility routes. Chat routes belong to LangGraph Agent Server.
- Keep public response metadata stable for frontend rendering: `citations`, `reranker_docs`, `context_usage`, `trace_id`, `mcp_used`, `mcp_tools_used`, and `mcp_tool_invocations`.
- Keep user-facing modes stable: `direct`, `rag`, `mcp`, and `mixed`.
- Keep `/api/config`, `/api/suggestions`, `/api/feedback`, `/api/documents/*`, and `/health` available through FastAPI.
- Preserve existing UI behavior with `@langchain/react` and `assistantId: "chat_agent"`.
- Use current LangGraph docs before changing LangGraph APIs or deployment config.
- Do not touch unrelated dirty worktree files unless the task explicitly requires it.

---

## File Structure

- Modify `src/rag_agent/graphs/state.py`: expand graph state so each node can pass structured execution fields without relying on `ChatRuntimeService` state snapshots.
- Create `src/rag_agent/graphs/runtime.py`: graph-specific helpers for extracting runtime context, thread id, run config, Langfuse trace context, and result/reference normalization.
- Modify `src/rag_agent/graphs/nodes/direct.py`: run direct LLM execution in the graph node.
- Modify `src/rag_agent/graphs/nodes/rag.py`: run contextualization, retrieval, reranking, and answer synthesis in the graph node.
- Modify `src/rag_agent/graphs/nodes/mcp.py`: run MCP-only execution in the graph node.
- Modify `src/rag_agent/graphs/nodes/mixed.py`: run mixed MCP plus retrieval-tool execution in the graph node.
- Modify `src/rag_agent/graphs/nodes/references.py`: keep metadata conversion small and graph-owned.
- Modify `src/rag_agent/runtime/chat_service.py`: shrink it after graph parity exists; keep only reusable helpers or non-LangGraph compatibility if still used by MCP server/tests.
- Modify `api/resources.py` and `api/deps/request.py`: remove FastAPI startup ownership of chat runtime if no FastAPI route needs it after the rewrite.
- Modify `docker-compose.yml`, `docker-compose.dev.yml`, `frontend/next.config.ts`, and frontend env docs only after API routes are verified through `http://localhost:2024`.
- Modify stale docs/scripts: `scripts/streaming_smoke_test.sh`, `observability/grafana/provisioning/dashboards/rag-api-pipeline.json`, `docs/CHAT_MEMORY_AND_SESSIONS.md`, `frontend/README.md`, and API docs references that still imply FastAPI owns chat streaming.
- Test with focused unit/workflow tests first, then Docker/Playwright smoke tests against the real Agent Server.

---

### Task 1: Lock Current Contracts Before Refactor

**Files:**
- Modify: `tests/workflow_tests/test_langgraph_server_bootstrap.py`
- Modify: `tests/workflow_tests/test_openapi_baseline.py`
- Create: `tests/workflow_tests/test_langgraph_chat_contract.py`
- Test: `tests/workflow_tests/test_langgraph_chat_contract.py`

**Interfaces:**
- Consumes: existing `chat_agent` graph from `src/rag_agent/graphs/chat_agent.py`
- Produces: regression tests that define the stable graph input, context, and assistant metadata contract

- [ ] **Step 1: Write contract tests for all modes**

Add `tests/workflow_tests/test_langgraph_chat_contract.py`:

```python
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.chat_agent import chat_agent


@pytest.mark.parametrize("mode", ["direct", "rag", "mcp", "mixed"])
def test_chat_agent_accepts_stable_context_modes(mode: str) -> None:
    graph_input = {"messages": [HumanMessage(content="Hello")]}
    config = {
        "configurable": {
            "thread_id": f"contract-{mode}",
            "mode": mode,
            "model_id": "fake-model",
            "session_id": "contract-session",
            "collection_name": "RAG_KNOWLEDGE_BASE",
            "enable_reranker": False,
            "enable_tracing": False,
            "mcp_server_keys": [],
        }
    }

    assert graph_input["messages"][0].content == "Hello"
    assert config["configurable"]["mode"] == mode


def test_assistant_metadata_shape_is_stable() -> None:
    message = AIMessage(
        content="Answer",
        additional_kwargs={
            "mode": "rag",
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
        },
    )

    assert message.additional_kwargs["mode"] == "rag"
    assert message.additional_kwargs["citations"] == []
    assert message.additional_kwargs["reranker_docs"] == []
    assert message.additional_kwargs["mcp_tools_used"] == []
```

- [ ] **Step 2: Run the new contract test**

Run: `uv run pytest tests/workflow_tests/test_langgraph_chat_contract.py -q`

Expected: PASS. This first test is a contract pin and should not require runtime changes.

- [ ] **Step 3: Assert FastAPI still has no `/api/langgraph/*` routes**

Keep or strengthen `tests/workflow_tests/test_openapi_baseline.py::test_custom_langgraph_routes_are_absent`:

```python
def test_custom_langgraph_routes_are_absent() -> None:
    with TestClient(app) as client:
        response = client.post("/api/langgraph/threads", json={})
    assert response.status_code == 404
```

- [ ] **Step 4: Run baseline route checks**

Run: `uv run pytest tests/workflow_tests/test_openapi_baseline.py tests/workflow_tests/test_langgraph_server_bootstrap.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/workflow_tests/test_langgraph_chat_contract.py tests/workflow_tests/test_openapi_baseline.py tests/workflow_tests/test_langgraph_server_bootstrap.py
git commit -m "test: lock langgraph chat and api route contracts"
```

---

### Task 2: Add Graph Runtime Helper Layer

**Files:**
- Create: `src/rag_agent/graphs/runtime.py`
- Modify: `src/rag_agent/graphs/nodes/references.py`
- Test: `tests/unit_tests/test_langgraph_runtime_helpers.py`

**Interfaces:**
- Consumes: `ChatGraphContext`, `Runtime[ChatGraphContext]`, `RunnableConfig`, result dictionaries from existing runtime helpers
- Produces:
  - `get_runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext`
  - `get_thread_id(runtime: Runtime[ChatGraphContext]) -> str | None`
  - `build_run_config(...) -> RunnableConfig`
  - `result_to_assistant_message(mode: str, result: dict[str, object]) -> AIMessage`

- [ ] **Step 1: Write failing helper tests**

Create `tests/unit_tests/test_langgraph_runtime_helpers.py`:

```python
from __future__ import annotations

from langchain_core.messages import AIMessage

from src.rag_agent.graphs.runtime import build_run_config, result_to_assistant_message


def test_build_run_config_places_chat_context_under_configurable() -> None:
    config = build_run_config(
        thread_id="thread-1",
        mode="rag",
        model_id="model-1",
        session_id="session-1",
        enable_tracing=True,
        mcp_server_keys=["oracle"],
        trace_context={"trace_id": "trace-1"},
    )

    configurable = config["configurable"]
    assert configurable["thread_id"] == "thread-1"
    assert configurable["mode"] == "rag"
    assert configurable["model_id"] == "model-1"
    assert configurable["session_id"] == "session-1"
    assert configurable["enable_tracing"] is True
    assert configurable["mcp_server_keys"] == ["oracle"]
    assert configurable["langfuse_trace_context"] == {"trace_id": "trace-1"}


def test_result_to_assistant_message_preserves_reference_payload() -> None:
    message = result_to_assistant_message(
        "mixed",
        {
            "final_answer": "Answer",
            "standalone_question": "Question",
            "citations": [{"source": "doc.md"}],
            "reranker_docs": [],
            "context_usage": {"chunks": 1},
            "mcp_used": True,
            "mcp_tools_used": ["lookup"],
            "mcp_tool_invocations": [{"tool_name": "lookup"}],
            "trace_id": "trace-1",
            "error": None,
        },
    )

    assert isinstance(message, AIMessage)
    assert message.content == "Answer"
    assert message.additional_kwargs["mode"] == "mixed"
    assert message.additional_kwargs["standalone_question"] == "Question"
    assert message.additional_kwargs["citations"] == [{"source": "doc.md"}]
    assert message.additional_kwargs["mcp_used"] is True
    assert message.additional_kwargs["mcp_tools_used"] == ["lookup"]
    assert message.additional_kwargs["trace_id"] == "trace-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit_tests/test_langgraph_runtime_helpers.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag_agent.graphs.runtime'`.

- [ ] **Step 3: Implement `src/rag_agent/graphs/runtime.py`**

```python
from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.state import ChatGraphContext


def get_runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return cast(ChatGraphContext, context)
    return {}


def get_thread_id(runtime: Runtime[ChatGraphContext]) -> str | None:
    thread_id = getattr(runtime.execution_info, "thread_id", None)
    return thread_id if isinstance(thread_id, str) and thread_id.strip() else None


def build_run_config(
    *,
    thread_id: str | None,
    mode: str,
    model_id: str | None,
    session_id: str | None,
    enable_tracing: bool | None,
    mcp_server_keys: list[str] | None,
    trace_context: dict[str, object] | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "mode": mode,
        "enable_tracing": bool(enable_tracing),
    }
    if thread_id:
        configurable["thread_id"] = thread_id
    if model_id:
        configurable["model_id"] = model_id
    if session_id:
        configurable["session_id"] = session_id
    if mcp_server_keys:
        configurable["mcp_server_keys"] = mcp_server_keys
    if trace_context:
        configurable["langfuse_trace_context"] = trace_context
    return cast(RunnableConfig, {"configurable": configurable})


def references_from_result(
    result: dict[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    references: dict[str, object] = {"mode": mode}
    for key in (
        "standalone_question",
        "citations",
        "reranker_docs",
        "context_usage",
        "trace_id",
        "mcp_used",
        "mcp_tools_used",
        "mcp_tool_invocations",
        "error",
    ):
        value = result.get(key)
        if value is None and key not in {"citations", "reranker_docs", "mcp_tools_used"}:
            continue
        if key in {"citations", "reranker_docs", "mcp_tools_used"} and not isinstance(value, list):
            references[key] = []
            continue
        references[key] = value
    return references


def result_to_assistant_message(mode: str, result: dict[str, object]) -> AIMessage:
    final_answer = result.get("final_answer")
    content = final_answer if isinstance(final_answer, str) else str(final_answer or "")
    references = references_from_result(result, mode=mode)
    return AIMessage(
        content=content,
        additional_kwargs=references,
        response_metadata=references,
    )
```

- [ ] **Step 4: Update `src/rag_agent/graphs/nodes/references.py` to delegate**

```python
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.rag_agent.graphs.runtime import result_to_assistant_message

logger = logging.getLogger(__name__)


def assistant_message_from_result(mode: str, result: dict[str, object]) -> AIMessage:
    return result_to_assistant_message(mode, result)


def assistant_message_from_exception(mode: str, exc: Exception) -> AIMessage:
    logger.exception("LangGraph %s node failed", mode)
    return result_to_assistant_message(
        mode,
        {
            "final_answer": (
                "I couldn't complete the request because the runtime backend "
                "returned an error. Please try again after the backend connection is healthy."
            ),
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        },
    )
```

- [ ] **Step 5: Run helper tests**

Run: `uv run pytest tests/unit_tests/test_langgraph_runtime_helpers.py -q`

Expected: PASS.

- [ ] **Step 6: Run graph contract tests**

Run: `uv run pytest tests/workflow_tests/test_langgraph_chat_contract.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rag_agent/graphs/runtime.py src/rag_agent/graphs/nodes/references.py tests/unit_tests/test_langgraph_runtime_helpers.py
git commit -m "refactor: add langgraph runtime helper layer"
```

---

### Task 3: Move Direct Mode Execution Into The Graph Node

**Files:**
- Modify: `src/rag_agent/graphs/nodes/direct.py`
- Test: `tests/unit_tests/test_langgraph_direct_node.py`

**Interfaces:**
- Consumes:
  - `get_runtime_context(runtime)`
  - `get_thread_id(runtime)`
  - `build_run_config(...)`
  - `result_to_assistant_message(...)`
  - `langchain_messages_to_dicts(...)`
- Produces: `run_direct_node(state, runtime) -> ChatGraphState` without calling `ChatRuntimeService.run_chat`

- [ ] **Step 1: Write failing direct node test**

Create `tests/unit_tests/test_langgraph_direct_node.py` with a fake runtime object and monkeypatched LLM call:

```python
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import direct


class FakeLlm:
    model_id = "fake-direct-model"


def test_direct_node_invokes_llm_without_chat_runtime_service(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fail_run_chat(*args: object, **kwargs: object) -> None:
        raise AssertionError("direct node must not call ChatRuntimeService.run_chat")

    def fake_get_llm(*, model_id: str | None = None) -> FakeLlm:
        calls.append({"model_id": model_id})
        return FakeLlm()

    def fake_invoke(llm: FakeLlm, history: list[Any], run_config: dict[str, object]) -> AIMessage:
        calls.append({"history": history, "run_config": run_config, "llm": llm})
        return AIMessage(content="Direct answer")

    monkeypatch.setattr(direct.ChatRuntimeService, "run_chat", fail_run_chat, raising=False)
    monkeypatch.setattr(direct, "get_llm", fake_get_llm)
    monkeypatch.setattr(direct, "invoke_llm_with_optional_config", fake_invoke)
    monkeypatch.setattr(
        direct,
        "emit_usage_observability",
        lambda **kwargs: (None, None),
    )

    runtime = SimpleNamespace(
        context={"model_id": "model-1", "session_id": "session-1", "enable_tracing": False},
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        direct.run_direct_node(
            {"messages": [HumanMessage(content="Hello")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Direct answer"
    assert assistant.additional_kwargs["mode"] == "direct"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/test_langgraph_direct_node.py -q`

Expected: FAIL because the current direct node calls `ChatRuntimeService().run_chat`.

- [ ] **Step 3: Implement direct node execution**

Replace `src/rag_agent/graphs/nodes/direct.py` with graph-local execution that imports:

```python
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.runtime.llm_invocation import invoke_llm_with_optional_config
from src.rag_agent.runtime.memory import langchain_messages_to_dicts
from src.rag_agent.utils.observability import emit_usage_observability
from src.rag_agent.utils.usage import extract_usage
```

The implementation must:

```python
messages = langchain_messages_to_dicts(state["messages"])
history = state["messages"]
context = get_runtime_context(runtime)
thread_id = get_thread_id(runtime)
run_cfg = build_run_config(...)
llm = get_llm(model_id=context.get("model_id"))
response = await asyncio.to_thread(invoke_llm_with_optional_config, llm, history, run_cfg)
usage = extract_usage(response)
emitted_usage, cost_usd = emit_usage_observability(...)
result = {
    "final_answer": str(getattr(response, "content", "") or "").strip(),
    "error": None,
    "standalone_question": latest_user_message(messages) or None,
    "citations": [],
    "reranker_docs": [],
    "context_usage": None,
    "mcp_used": False,
    "mcp_tools_used": [],
    "model_id": getattr(llm, "model_id", None) or context.get("model_id"),
    "usage": emitted_usage,
    "cost_usd": cost_usd,
}
assistant_message = assistant_message_from_result("direct", result)
return {"messages": [assistant_message], "references": assistant_message.additional_kwargs}
```

- [ ] **Step 4: Run direct node test**

Run: `uv run pytest tests/unit_tests/test_langgraph_direct_node.py -q`

Expected: PASS.

- [ ] **Step 5: Run direct mode service regression tests**

Run: `uv run pytest tests/unit_tests/test_graph_service_oci_runtime.py -k direct_mode -q`

Expected: PASS. If failures occur, keep `ChatRuntimeService` direct mode behavior intact until all graph callers are migrated.

- [ ] **Step 6: Commit**

```bash
git add src/rag_agent/graphs/nodes/direct.py tests/unit_tests/test_langgraph_direct_node.py
git commit -m "refactor: run direct mode inside langgraph node"
```

---

### Task 4: Move RAG Mode Execution Into The Graph Node

**Files:**
- Modify: `src/rag_agent/graphs/nodes/rag.py`
- Test: `tests/unit_tests/test_langgraph_rag_node.py`

**Interfaces:**
- Consumes: graph runtime helpers, `contextualize_question`, `rag_runtime.retrieve_oracle_docs`, `rag_runtime.rerank_retrieved_docs`, `rag_runtime.synthesize_rag_answer`, `rag_runtime.citations_from_docs`, `rag_runtime.serialize_docs`
- Produces: `run_rag_node(state, runtime) -> ChatGraphState` without calling `ChatRuntimeService.run_chat`

- [ ] **Step 1: Write failing RAG node test**

Create `tests/unit_tests/test_langgraph_rag_node.py`:

```python
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import rag


def test_rag_node_retrieves_and_returns_citations_without_chat_runtime_service(monkeypatch) -> None:
    docs = [Document(page_content="Payment is due in 45 days.", metadata={"source": "terms.md"})]

    def fail_run_chat(*args: object, **kwargs: object) -> None:
        raise AssertionError("rag node must not call ChatRuntimeService.run_chat")

    async def fake_contextualize_question(**kwargs: object) -> str:
        return "What are the payment terms?"

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        return "Payment is due in 45 days.", None, "fake-rag-model"

    monkeypatch.setattr(rag.ChatRuntimeService, "run_chat", fail_run_chat, raising=False)
    monkeypatch.setattr(rag, "contextualize_question", fake_contextualize_question)
    monkeypatch.setattr(rag.rag_runtime, "retrieve_oracle_docs", lambda **kwargs: docs)
    monkeypatch.setattr(rag.rag_runtime, "rerank_retrieved_docs", lambda query, docs, *, enable_reranker: docs)
    monkeypatch.setattr(rag.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(rag.rag_runtime, "citations_from_docs", lambda docs: [{"source": "terms.md"}])
    monkeypatch.setattr(rag.rag_runtime, "serialize_docs", lambda docs: [{"source": "terms.md", "content": "Payment is due in 45 days."}])
    monkeypatch.setattr(rag, "emit_usage_observability", lambda **kwargs: (None, None))

    runtime = SimpleNamespace(
        context={
            "model_id": "model-1",
            "session_id": "session-1",
            "collection_name": "kb",
            "enable_reranker": False,
            "enable_tracing": False,
        },
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        rag.run_rag_node(
            {"messages": [HumanMessage(content="What are the payment terms?")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Payment is due in 45 days."
    assert assistant.additional_kwargs["mode"] == "rag"
    assert assistant.additional_kwargs["citations"] == [{"source": "terms.md"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/test_langgraph_rag_node.py -q`

Expected: FAIL because the current RAG node calls `ChatRuntimeService().run_chat`.

- [ ] **Step 3: Implement RAG node execution**

Move the RAG path from `ChatRuntimeService._run_rag_mode` into `run_rag_node`, preserving:

```python
question = latest_user_message(messages)
chat_history = chat_history_before_latest_user(messages)
standalone_question = await contextualize_question(...)
docs = await asyncio.to_thread(rag_runtime.retrieve_oracle_docs, ...)
docs = await asyncio.to_thread(rag_runtime.rerank_retrieved_docs, ...)
if docs:
    rag_answer, rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(...)
else:
    rag_answer = _NO_ORACLE_CONTEXT_ANSWER
    rag_usage = None
    resolved_model_id = context.get("model_id") or "unknown"
```

Return assistant metadata with `citations`, `reranker_docs`, `standalone_question`, `usage`, `cost_usd`, `model_id`, and `mode`.

- [ ] **Step 4: Run RAG node tests**

Run: `uv run pytest tests/unit_tests/test_langgraph_rag_node.py -q`

Expected: PASS.

- [ ] **Step 5: Run RAG service regression tests**

Run: `uv run pytest tests/unit_tests/test_graph_service_oci_runtime.py -k rag_mode -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rag_agent/graphs/nodes/rag.py tests/unit_tests/test_langgraph_rag_node.py
git commit -m "refactor: run rag mode inside langgraph node"
```

---

### Task 5: Move MCP And Mixed Execution Into Graph Nodes

**Files:**
- Modify: `src/rag_agent/graphs/nodes/mcp.py`
- Modify: `src/rag_agent/graphs/nodes/mixed.py`
- Test: `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`

**Interfaces:**
- Consumes: `run_mcp_agent_turn`, `tool_failure_summary`, retrieval tool builder currently in `ChatRuntimeService._build_oracle_retrieval_tool`
- Produces:
  - MCP node result metadata: `mcp_used`, `mcp_tools_used`, `mcp_tool_invocations`
  - Mixed node result metadata: retrieval citations when retrieval tool state exists, MCP metadata when tools are used

- [ ] **Step 1: Write failing MCP and mixed node tests**

Create `tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import mcp, mixed


@dataclass
class FakeMcpTurn:
    answer: str = "Tool answer"
    tools_used: list[str] = None  # type: ignore[assignment]
    tool_invocations: list[dict[str, object]] = None  # type: ignore[assignment]
    resolved_model_id: str = "fake-mcp-model"

    def __post_init__(self) -> None:
        if self.tools_used is None:
            self.tools_used = ["lookup"]
        if self.tool_invocations is None:
            self.tool_invocations = [{"tool_name": "lookup", "status": "success"}]


def test_mcp_node_runs_agent_turn_without_chat_runtime_service(monkeypatch) -> None:
    def fail_run_chat(*args: object, **kwargs: object) -> None:
        raise AssertionError("mcp node must not call ChatRuntimeService.run_chat")

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        return FakeMcpTurn()

    monkeypatch.setattr(mcp.ChatRuntimeService, "run_chat", fail_run_chat, raising=False)
    monkeypatch.setattr(mcp, "run_mcp_agent_turn", fake_run_mcp_agent_turn)

    runtime = SimpleNamespace(
        context={"model_id": "model-1", "session_id": "session-1", "enable_tracing": False},
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        mcp.run_mcp_node(
            {"messages": [HumanMessage(content="Use a tool")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Tool answer"
    assert assistant.additional_kwargs["mode"] == "mcp"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["mcp_tools_used"] == ["lookup"]


def test_mixed_node_runs_agent_turn_without_chat_runtime_service(monkeypatch) -> None:
    def fail_run_chat(*args: object, **kwargs: object) -> None:
        raise AssertionError("mixed node must not call ChatRuntimeService.run_chat")

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        return FakeMcpTurn(answer="Mixed answer")

    monkeypatch.setattr(mixed.ChatRuntimeService, "run_chat", fail_run_chat, raising=False)
    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(mixed, "emit_usage_observability", lambda **kwargs: (None, None))

    runtime = SimpleNamespace(
        context={
            "model_id": "model-1",
            "session_id": "session-1",
            "collection_name": "kb",
            "enable_reranker": False,
            "enable_tracing": False,
        },
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        mixed.run_mixed_node(
            {"messages": [HumanMessage(content="Use tools and docs")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Mixed answer"
    assert assistant.additional_kwargs["mode"] == "mixed"
    assert assistant.additional_kwargs["mcp_used"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit_tests/test_langgraph_mcp_mixed_nodes.py -q`

Expected: FAIL because current MCP and mixed nodes call `ChatRuntimeService().run_chat`.

- [ ] **Step 3: Extract retrieval-tool builder if mixed needs it**

If `ChatRuntimeService._build_oracle_retrieval_tool` is still private, move its reusable logic to:

```python
src/rag_agent/runtime/retrieval_tool.py
```

Expose:

```python
def build_oracle_retrieval_tool(collection_name: str | None) -> object:
    ...
```

Update `ChatRuntimeService` and `mixed.py` to import this helper so behavior stays identical during migration.

- [ ] **Step 4: Implement MCP node execution**

Move the MCP body from `ChatRuntimeService._run_mcp_mode` into `run_mcp_node`, preserving:

```python
question = latest_user_message(messages)
chat_history = chat_history_before_latest_user(messages)
resolved_model_id = context.get("model_id") or get_llm().model_id
run_cfg = build_run_config(...)
mcp_turn = await run_mcp_agent_turn(...)
result = {
    "final_answer": mcp_turn.answer,
    "error": None,
    "standalone_question": question or None,
    "citations": [],
    "reranker_docs": [],
    "context_usage": None,
    "mcp_used": bool(mcp_turn.tools_used),
    "mcp_tools_used": mcp_turn.tools_used,
    "mcp_tool_invocations": mcp_turn.tool_invocations,
    "model_id": mcp_turn.resolved_model_id,
}
```

- [ ] **Step 5: Implement mixed node execution**

Move the mixed body from `ChatRuntimeService._run_mixed_mode` into `run_mixed_node`, preserving:

```python
retrieval_tool = build_oracle_retrieval_tool(context.get("collection_name"))
mcp_turn = await run_mcp_agent_turn(..., extra_tools=[retrieval_tool] or equivalent current API)
result = {
    "final_answer": mcp_turn.answer,
    "error": None,
    "standalone_question": question or None,
    "citations": existing retrieval citations from current mixed implementation,
    "reranker_docs": existing serialized docs from current mixed implementation,
    "context_usage": existing context usage value,
    "mcp_used": bool(mcp_turn.tools_used),
    "mcp_tools_used": mcp_turn.tools_used,
    "mcp_tool_invocations": mcp_turn.tool_invocations,
    "model_id": mcp_turn.resolved_model_id,
}
```

- [ ] **Step 6: Run MCP/mixed node tests**

Run: `uv run pytest tests/unit_tests/test_langgraph_mcp_mixed_nodes.py -q`

Expected: PASS.

- [ ] **Step 7: Run MCP/mixed service regression tests**

Run: `uv run pytest tests/unit_tests/test_graph_service_oci_runtime.py -k "mcp_mode or mixed_mode" -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rag_agent/graphs/nodes/mcp.py src/rag_agent/graphs/nodes/mixed.py src/rag_agent/runtime/retrieval_tool.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py
git commit -m "refactor: run mcp and mixed modes inside langgraph nodes"
```

---

### Task 6: Shrink Or Retire ChatRuntimeService From Chat Ownership

**Files:**
- Modify: `src/rag_agent/runtime/chat_service.py`
- Modify: `api/resources.py`
- Modify: `api/deps/request.py`
- Modify: `mcp_servers/mcp_rag_server.py`
- Test: `tests/unit_tests/test_api_resources.py`
- Test: `tests/unit_tests/test_runtime_agent.py`
- Test: `tests/unit_tests/test_graph_service_oci_runtime.py`

**Interfaces:**
- Consumes: graph nodes now own mode execution
- Produces: no FastAPI app startup dependency on `ChatRuntimeService` unless a non-chat API route still needs it

- [ ] **Step 1: Identify remaining callers structurally**

Run: use CodeGraph `codegraph_callers` for `ChatRuntimeService` and `run_chat`.

Expected: graph nodes no longer call `ChatRuntimeService.run_chat`; remaining callers are tests, MCP server compatibility, or legacy runtime stream helpers.

- [ ] **Step 2: Write a resource test that FastAPI no longer builds chat runtime**

Update `tests/unit_tests/test_api_resources.py` with:

```python
def test_app_resources_do_not_require_chat_runtime_for_product_api_startup(monkeypatch) -> None:
    from api import resources

    def fail_chat_runtime(*args: object, **kwargs: object) -> None:
        raise AssertionError("FastAPI product API startup must not construct chat runtime")

    monkeypatch.setattr(resources, "ChatRuntimeService", fail_chat_runtime)

    app_resources = asyncio.run(resources.create_app_resources())

    assert app_resources.settings is not None
```

- [ ] **Step 3: Run resource test to verify it fails**

Run: `uv run pytest tests/unit_tests/test_api_resources.py -q`

Expected: FAIL while `create_app_resources()` still constructs `ChatRuntimeService`.

- [ ] **Step 4: Modify `AppResources`**

Change `api/resources.py` so `AppResources` only owns:

```python
@dataclass
class AppResources:
    settings: Settings
    _state_conn: object | None = None
```

Only keep a chat runtime field if a FastAPI route still uses `get_graph_service`.

- [ ] **Step 5: Remove or isolate `get_graph_service` fallback**

If no route uses it, delete `get_graph_service` from `api/deps/request.py`. If `mcp_servers/mcp_rag_server.py` still needs a service, construct the needed compatibility runtime there instead of app startup.

- [ ] **Step 6: Shrink `ChatRuntimeService`**

Keep `ChatRuntimeService` only if `astream_events`, `get_state_values`, or MCP server compatibility still requires it. Otherwise split reusable functions into focused modules and leave a short compatibility class with clear deprecation comments.

- [ ] **Step 7: Run runtime tests**

Run: `uv run pytest tests/unit_tests/test_api_resources.py tests/unit_tests/test_runtime_agent.py tests/unit_tests/test_graph_service_oci_runtime.py -q`

Expected: PASS or a small set of failures that identify remaining compatibility callers. Do not delete compatibility paths until tests prove they are unused or replaced.

- [ ] **Step 8: Commit**

```bash
git add api/resources.py api/deps/request.py src/rag_agent/runtime/chat_service.py mcp_servers/mcp_rag_server.py tests/unit_tests/test_api_resources.py tests/unit_tests/test_runtime_agent.py tests/unit_tests/test_graph_service_oci_runtime.py
git commit -m "refactor: remove fastapi ownership of chat runtime"
```

---

### Task 7: Prove Product APIs Work Through LangGraph Agent Server

**Files:**
- Create: `tests/integration_tests/test_langgraph_custom_http_app_live.py`
- Modify: `scripts/streaming_smoke_test.sh`
- Test: live LangGraph Agent Server on `http://127.0.0.1:2024`

**Interfaces:**
- Consumes: `langgraph.json` with `"http": {"app": "./api/main.py:app"}`
- Produces: evidence that `/api/config`, `/api/suggestions`, `/api/documents/*`, `/api/feedback`, and `/health` are reachable through port `2024`

- [ ] **Step 1: Write gated live test**

Create `tests/integration_tests/test_langgraph_custom_http_app_live.py`:

```python
from __future__ import annotations

import os

import pytest
import requests


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LANGGRAPH_LIVE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_LIVE_TESTS=1 with LangGraph Agent Server running",
)


def _base_url() -> str:
    return os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")


def test_fastapi_custom_routes_are_available_from_langgraph_server() -> None:
    base = _base_url()

    health = requests.get(f"{base}/health", timeout=10)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    config = requests.get(f"{base}/api/config", timeout=20)
    assert config.status_code == 200
    body = config.json()
    assert "model_list" in body
    assert "collection_list" in body
```

- [ ] **Step 2: Run without env flag**

Run: `uv run pytest tests/integration_tests/test_langgraph_custom_http_app_live.py -q`

Expected: SKIPPED.

- [ ] **Step 3: Start local stack**

Run: `docker compose up -d langgraph`

Expected: `rag-langgraph` is healthy or starting.

- [ ] **Step 4: Run live custom-route test**

Run: `RUN_LANGGRAPH_LIVE_TESTS=1 uv run pytest tests/integration_tests/test_langgraph_custom_http_app_live.py -q`

Expected: PASS.

- [ ] **Step 5: Update streaming smoke test to use `2024`**

Change `scripts/streaming_smoke_test.sh` defaults:

```bash
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-2024}"
API_URL="http://${API_HOST}:${API_PORT}"
```

Change request path:

```bash
curl -N -X POST "${API_URL}/threads/smoke-thread/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"chat_agent","input":{"messages":[{"role":"user","content":"Hello"}]},"config":{"configurable":{"mode":"direct","enable_tracing":false}}}'
```

- [ ] **Step 6: Run smoke test against LangGraph**

Run: `scripts/streaming_smoke_test.sh`

Expected: PASS with `Streaming smoke test passed`.

- [ ] **Step 7: Commit**

```bash
git add tests/integration_tests/test_langgraph_custom_http_app_live.py scripts/streaming_smoke_test.sh
git commit -m "test: prove custom routes through langgraph server"
```

---

### Task 8: Consolidate Frontend Backend Bases After Live Proof

**Files:**
- Modify: `frontend/next.config.ts`
- Modify: `frontend/src/lib/api-base.ts`
- Modify: `frontend/src/hooks/chat/stream-config.ts`
- Modify: `frontend/env.example`
- Test: `frontend/src/hooks/chat/__tests__/stream-config.test.ts`
- Test: add `frontend/src/lib/__tests__/api-base.test.ts`

**Interfaces:**
- Consumes: product APIs verified on `http://localhost:2024`
- Produces: one frontend backend origin for both `/api/*` and LangGraph streaming, with optional separate override retained only if necessary

- [ ] **Step 1: Add API base tests**

Create `frontend/src/lib/__tests__/api-base.test.ts`:

```typescript
import { describe, expect, it, vi, afterEach } from "vitest";
import { getClientApiBase, toApiUrl } from "@/lib/api-base";

describe("api-base", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("prefers NEXT_PUBLIC_API_BASE when set", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "http://localhost:2024/");
    expect(getClientApiBase()).toBe("http://localhost:2024");
    expect(toApiUrl("/api/config")).toBe("http://localhost:2024/api/config");
  });

  it("falls back to LangGraph backend base on the server", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "");
    vi.stubEnv("LANGGRAPH_BACKEND_URL", "http://langgraph:2024/");
    expect(getClientApiBase()).toBe("http://langgraph:2024");
  });
});
```

- [ ] **Step 2: Run frontend unit tests to verify failure if current fallback differs**

Run: `pnpm --dir frontend test src/lib/__tests__/api-base.test.ts src/hooks/chat/__tests__/stream-config.test.ts`

Expected: FAIL if `api-base.ts` still defaults to FastAPI `3002`.

- [ ] **Step 3: Update frontend API defaults**

Change `frontend/src/lib/api-base.ts` fallback to prefer the LangGraph host:

```typescript
return normalizeBase(
  process.env.LANGGRAPH_BACKEND_URL ||
  process.env.NEXT_PUBLIC_LANGGRAPH_API_BASE ||
  "http://localhost:2024",
);
```

Keep `NEXT_PUBLIC_API_BASE` as the explicit browser override.

- [ ] **Step 4: Update Next rewrites**

Change `frontend/next.config.ts` so `/api/:path*` defaults to the same LangGraph backend:

```typescript
const langgraphUrl =
  process.env.LANGGRAPH_BACKEND_URL ||
  process.env.NEXT_PUBLIC_LANGGRAPH_API_BASE ||
  "http://localhost:2024";
const backendUrl = process.env.FASTAPI_BACKEND_URL || langgraphUrl;
```

This keeps `FASTAPI_BACKEND_URL` as an escape hatch while making LangGraph the default.

- [ ] **Step 5: Update env example**

Set:

```env
NEXT_PUBLIC_API_BASE=http://localhost:2024
NEXT_PUBLIC_LANGGRAPH_API_BASE=http://localhost:2024
LANGGRAPH_BACKEND_URL=http://localhost:2024
```

- [ ] **Step 6: Run frontend tests**

Run: `pnpm --dir frontend test src/lib/__tests__/api-base.test.ts src/hooks/chat/__tests__/stream-config.test.ts`

Expected: PASS.

- [ ] **Step 7: Run frontend build**

Run: `pnpm --dir frontend build`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend/next.config.ts frontend/src/lib/api-base.ts frontend/src/hooks/chat/stream-config.ts frontend/env.example frontend/src/lib/__tests__/api-base.test.ts
git commit -m "refactor: default frontend api traffic to langgraph server"
```

---

### Task 9: Remove Separate Backend Service Only After Consolidation Is Proven

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker-compose.dev.yml`
- Modify: `docker/Dockerfile`
- Modify: `README.md`
- Modify: `docs/DOCKER-SETUP.md`
- Modify: `docs/GETTING-STARTED.md`
- Modify: `docs/CONFIGURATION.md`
- Test: Docker Compose stack

**Interfaces:**
- Consumes: Task 7 proof that FastAPI routes work from LangGraph server, Task 8 frontend base consolidation
- Produces: one backend service in Docker for app runtime: `langgraph`

- [ ] **Step 1: Change Compose frontend dependency**

In `docker-compose.yml`, remove frontend dependency on `backend` and keep:

```yaml
depends_on:
  langgraph:
    condition: service_healthy
```

- [ ] **Step 2: Remove or profile-gate standalone backend**

Move the `backend` service behind a profile:

```yaml
profiles:
  - legacy-fastapi
```

Do not delete the Dockerfile target in the same task; keep rollback easy.

- [ ] **Step 3: Update frontend Docker args**

Set:

```yaml
NEXT_PUBLIC_API_BASE: ${NEXT_PUBLIC_API_BASE:-http://localhost:2024}
NEXT_PUBLIC_LANGGRAPH_API_BASE: ${NEXT_PUBLIC_LANGGRAPH_API_BASE:-http://localhost:2024}
LANGGRAPH_BACKEND_URL: ${LANGGRAPH_BACKEND_URL:-http://langgraph:2024}
FASTAPI_BACKEND_URL: ${FASTAPI_BACKEND_URL:-http://langgraph:2024}
```

- [ ] **Step 4: Update docs**

Replace “Backend (FastAPI) on `3002`” as the default with:

```text
LangGraph Agent Server: http://localhost:2024
Product API routes: http://localhost:2024/api/*
Optional legacy FastAPI backend profile: http://localhost:3002
```

- [ ] **Step 5: Run Compose build**

Run: `docker compose build langgraph frontend`

Expected: PASS.

- [ ] **Step 6: Start runtime**

Run: `docker compose up -d langgraph frontend`

Expected: `rag-langgraph` and `rag-frontend` healthy/running.

- [ ] **Step 7: Verify routes**

Run:

```bash
curl -s http://127.0.0.1:2024/health
curl -s http://127.0.0.1:2024/api/config
```

Expected: health returns `{"status":"ok"}` and config returns JSON with `model_list`.

- [ ] **Step 8: Run focused E2E**

Run: `pnpm --dir frontend test:e2e`

Expected: PASS, or report exact failures and keep backend profile as default until fixed.

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml docker-compose.dev.yml docker/Dockerfile README.md docs/DOCKER-SETUP.md docs/GETTING-STARTED.md docs/CONFIGURATION.md
git commit -m "chore: consolidate docker runtime on langgraph server"
```

---

### Task 10: Clean Stale Docs, Metrics, And Changelog

**Files:**
- Modify: `observability/grafana/provisioning/dashboards/rag-api-pipeline.json`
- Modify: `docs/CHAT_MEMORY_AND_SESSIONS.md`
- Modify: `docs/CHAT_STREAMING_PROTOCOL.md`
- Modify: `docs/MCP-USAGE.md`
- Modify: `frontend/README.md`
- Modify: `docs/api/20-chat/README.md`
- Modify: `CHANGELOG.md`
- Test: docs and API docs checks

**Interfaces:**
- Consumes: final runtime topology
- Produces: docs and metrics that no longer mention active `/api/langgraph/*` chat ownership

- [ ] **Step 1: Update Grafana dashboard queries**

Replace log filters containing:

```text
/api/langgraph/threads/
```

with filters matching current LangGraph traffic:

```text
/threads/
/runs/stream
```

Keep service labels aligned with whichever container emits the logs after Task 9.

- [ ] **Step 2: Update chat memory docs**

Replace references to:

```text
api/routes/langgraph_server.py
DELETE /api/threads/{thread_id}
```

with:

```text
LangGraph Agent Server thread APIs
DELETE /threads/{thread_id}
```

- [ ] **Step 3: Update frontend README**

State that the UI talks to:

```text
NEXT_PUBLIC_LANGGRAPH_API_BASE for chat/thread streaming.
NEXT_PUBLIC_API_BASE for product APIs, defaulting to the same LangGraph server.
```

- [ ] **Step 4: Add changelog entry**

Under `CHANGELOG.md` current date, add:

```markdown
- Planned graph-native chat runtime migration: LangGraph Agent Server remains the chat/thread/run owner, FastAPI stays mounted as `http.app` for product APIs, and stale `/api/langgraph/*` ownership is being removed from scripts, docs, and dashboards.
```

- [ ] **Step 5: Run docs/API checks**

Run: `uv run python scripts/sync_api_docs.py --check`

Expected: PASS.

Run: `uv run pytest tests/workflow_tests/test_api_docs_sync.py tests/workflow_tests/test_openapi_baseline.py -q`

Expected: PASS.

- [ ] **Step 6: Run stale-route grep**

Run: `rg -n "/api/langgraph|api/routes/langgraph_server.py|3002.*chat|FastAPI.*chat streaming" docs README.md frontend scripts observability`

Expected: only intentional history/changelog references remain.

- [ ] **Step 7: Commit**

```bash
git add observability/grafana/provisioning/dashboards/rag-api-pipeline.json docs/CHAT_MEMORY_AND_SESSIONS.md docs/CHAT_STREAMING_PROTOCOL.md docs/MCP-USAGE.md frontend/README.md docs/api/20-chat/README.md CHANGELOG.md
git commit -m "docs: align chat docs with langgraph ownership"
```

---

## Final Verification

- [ ] Run Python focused tests:

```bash
uv run pytest tests/unit_tests/test_langgraph_runtime_helpers.py tests/unit_tests/test_langgraph_direct_node.py tests/unit_tests/test_langgraph_rag_node.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py -q
```

- [ ] Run workflow tests:

```bash
uv run pytest tests/workflow_tests/test_langgraph_chat_contract.py tests/workflow_tests/test_langgraph_server_bootstrap.py tests/workflow_tests/test_openapi_baseline.py -q
```

- [ ] Run frontend checks:

```bash
pnpm --dir frontend build
pnpm --dir frontend test:e2e
```

- [ ] Run Docker smoke:

```bash
docker compose up -d langgraph frontend
scripts/streaming_smoke_test.sh
curl -s http://127.0.0.1:2024/health
curl -s http://127.0.0.1:2024/api/config
```

- [ ] Inspect Langfuse/OTEL traces for one live chat turn from `rag-langgraph`.

---

## Self-Review

- Spec coverage: The plan keeps FastAPI as `http.app`, moves chat execution into graph-owned nodes, removes stale `/api/langgraph/*` assumptions, and only consolidates Docker/frontend bases after live proof.
- Placeholder scan: No placeholder markers or intentionally vague implementation steps remain.
- Type consistency: Helper names from Task 2 are reused consistently in later node tasks.
- Scope check: This is one coherent migration, but Tasks 1-6 can ship without Tasks 8-9 if deployment consolidation is deferred.

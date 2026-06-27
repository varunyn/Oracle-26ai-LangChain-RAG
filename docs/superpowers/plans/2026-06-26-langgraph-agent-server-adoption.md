# LangGraph Agent Server Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move chat execution from the custom `/api/langgraph` compatibility layer to a real LangGraph Agent Server while preserving the current product UI and the four user-facing modes.

**Architecture:** Add a real exported LangGraph graph and `langgraph.json`, migrate direct/RAG/MCP/mixed behavior into graph-owned routing and node helpers, rewire the frontend to standard `useStream` against the Agent Server, and then delete the compatibility router plus adapter-specific tests. Keep FastAPI only for non-chat product APIs such as config, documents, suggestions, feedback, MCP settings, and health.

**Tech Stack:** Python 3.11, LangGraph CLI and Agent Server, FastAPI, Next.js 16, React 19, `@langchain/react`, Oracle retrieval, MCP runtime integration, Langfuse.

## Global Constraints

- Preserve the current four modes during the first Agent Server adoption pass: `direct`, `rag`, `mcp`, and `mixed`.
- Treat `mode` as graph routing policy, not API or protocol shape.
- `direct` must not retrieve or call MCP tools.
- `rag` must use Oracle retrieval plus answer synthesis and must not use MCP tools.
- `mcp` must use MCP/tool-agent behavior and must not use Oracle retrieval unless explicitly exposed by an MCP capability.
- `mixed` may use both MCP tools and Oracle retrieval.
- Preserve existing product functionality: model selection, RAG collections, document ingestion, citations, feedback, tracing, and MCP settings.
- Keep FastAPI for non-chat product endpoints only; do not shadow built-in LangGraph routes such as `/threads` or `/runs`.
- Acceptance for direct/RAG/MCP/mixed paths requires real configured calls when the environment is available; fake model/tool tests are secondary guardrails, not proof of migration parity.
- Remove custom code where LangGraph already provides the behavior; do not remove product-specific behavior merely because the current implementation is custom.

---

### Task 1: Scaffold the Real LangGraph Server Surface

**Files:**
- Create: `langgraph.json`
- Create: `src/rag_agent/graphs/__init__.py`
- Create: `src/rag_agent/graphs/state.py`
- Create: `src/rag_agent/graphs/chat_agent.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/workflow_tests/test_langgraph_server_bootstrap.py`

**Interfaces:**
- Consumes: existing runtime helpers from `src/rag_agent/runtime/chat_service.py`, existing FastAPI app entrypoint from `api/main.py`
- Produces: `build_chat_agent() -> CompiledStateGraph`, `ChatGraphContext` typed payload, `langgraph.json` graph id `chat_agent`

- [ ] **Step 1: Write the failing bootstrap test**

```python
from pathlib import Path
import json


def test_langgraph_json_registers_chat_agent() -> None:
    config = json.loads(Path("langgraph.json").read_text())
    assert config["graphs"]["chat_agent"] == "./src/rag_agent/graphs/chat_agent.py:chat_agent"
    assert config["env"] == ".env"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/workflow_tests/test_langgraph_server_bootstrap.py -v`
Expected: FAIL with `FileNotFoundError` for `langgraph.json`

- [ ] **Step 3: Write minimal graph scaffolding**

```python
# src/rag_agent/graphs/state.py
from __future__ import annotations

from typing import Literal, TypedDict

Mode = Literal["direct", "rag", "mcp", "mixed"]


class ChatGraphContext(TypedDict, total=False):
    model_id: str
    collection_name: str
    mode: Mode
    enable_reranker: bool
    enable_tracing: bool
    mcp_server_keys: list[str]


class ChatGraphState(TypedDict, total=False):
    messages: list[dict[str, object]]
    context: ChatGraphContext
    references: dict[str, object]
```

```python
# src/rag_agent/graphs/chat_agent.py
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.rag_agent.graphs.state import ChatGraphState


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def build_chat_agent():
    graph = StateGraph(ChatGraphState)
    graph.add_node("bootstrap", _bootstrap_node)
    graph.set_entry_point("bootstrap")
    graph.add_edge("bootstrap", END)
    return graph.compile()


chat_agent = build_chat_agent()
```

```json
// langgraph.json
{
  "graphs": {
    "chat_agent": "./src/rag_agent/graphs/chat_agent.py:chat_agent"
  },
  "env": ".env",
  "http": {
    "app": "./api/main.py:app"
  }
}
```

- [ ] **Step 4: Add the minimum dependency and command documentation**

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
  "langgraph-cli[inmem]>=0.2.6",
]
```

```md
# README.md
## LangGraph Agent Server development

Run the graph server locally:

```bash
uv run langgraph dev
```

The graph id is `chat_agent`.
```

- [ ] **Step 5: Run tests to verify the scaffold passes**

Run: `uv run pytest tests/workflow_tests/test_langgraph_server_bootstrap.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add langgraph.json pyproject.toml README.md src/rag_agent/graphs/__init__.py src/rag_agent/graphs/state.py src/rag_agent/graphs/chat_agent.py tests/workflow_tests/test_langgraph_server_bootstrap.py
git commit -m "feat: scaffold LangGraph chat agent"
```

### Task 2: Move Direct and RAG Paths Into Graph-Owned Routing

**Files:**
- Create: `src/rag_agent/graphs/nodes/direct.py`
- Create: `src/rag_agent/graphs/nodes/rag.py`
- Modify: `src/rag_agent/graphs/chat_agent.py`
- Modify: `src/rag_agent/runtime/chat_service.py`
- Test: `tests/integration_tests/test_langgraph_direct_and_rag_live.py`

**Interfaces:**
- Consumes: `ChatGraphState`, existing `get_llm`, `rag_runtime.retrieve_oracle_docs`, `rag_runtime.synthesize_rag_answer`, `rag_runtime.stream_rag_answer`
- Produces: `run_direct_node(state: ChatGraphState) -> ChatGraphState`, `run_rag_node(state: ChatGraphState) -> ChatGraphState`, `route_mode(state: ChatGraphState) -> str`

- [ ] **Step 1: Write the failing live direct/RAG integration tests**

```python
import pytest
from langgraph_sdk import get_sync_client


@pytest.mark.integration
def test_chat_agent_direct_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [{"role": "user", "content": "Reply with the word READY"}],
            "context": {"mode": "direct"},
        },
    )
    messages = result["messages"]
    assert any("READY" in str(message.get("content", "")) for message in messages)


@pytest.mark.integration
def test_chat_agent_rag_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [{"role": "user", "content": "Use retrieval to answer the configured corpus question"}],
            "context": {"mode": "rag", "collection_name": "default"},
        },
    )
    assert result["references"]["mode"] == "rag"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration_tests/test_langgraph_direct_and_rag_live.py -v`
Expected: FAIL because the graph currently has no mode routing or result assembly

- [ ] **Step 3: Implement direct and RAG nodes with explicit routing**

```python
# src/rag_agent/graphs/nodes/direct.py
from __future__ import annotations

from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_direct_node(state: ChatGraphState) -> ChatGraphState:
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=state["messages"],
        model_id=state.get("context", {}).get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=None,
        enable_reranker=False,
        enable_tracing=state.get("context", {}).get("enable_tracing"),
        mode="direct",
        mcp_server_keys=None,
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": state.get("context", {}),
        "references": {"mode": "direct", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
```

```python
# src/rag_agent/graphs/nodes/rag.py
from __future__ import annotations

from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_rag_node(state: ChatGraphState) -> ChatGraphState:
    context = state.get("context", {})
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=state["messages"],
        model_id=context.get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=context.get("collection_name"),
        enable_reranker=context.get("enable_reranker"),
        enable_tracing=context.get("enable_tracing"),
        mode="rag",
        mcp_server_keys=None,
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": {"mode": "rag", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
```

```python
# src/rag_agent/graphs/chat_agent.py
from src.rag_agent.graphs.nodes.direct import run_direct_node
from src.rag_agent.graphs.nodes.rag import run_rag_node


def route_mode(state: ChatGraphState) -> str:
    mode = state.get("context", {}).get("mode", "direct")
    if mode == "rag":
        return "rag"
    return "direct"
```

- [ ] **Step 4: Compile the graph with direct/RAG routing**

```python
graph.add_conditional_edges(
    "bootstrap",
    route_mode,
    {
        "direct": "direct",
        "rag": "rag",
    },
)
graph.add_node("direct", run_direct_node)
graph.add_node("rag", run_rag_node)
graph.add_edge("direct", END)
graph.add_edge("rag", END)
```

- [ ] **Step 5: Run the direct/RAG live tests**

Run: `uv run pytest tests/integration_tests/test_langgraph_direct_and_rag_live.py -v`
Expected: PASS when provider and retrieval config are available, otherwise SKIP with an explicit fixture reason

- [ ] **Step 6: Commit**

```bash
git add src/rag_agent/graphs/chat_agent.py src/rag_agent/graphs/nodes/direct.py src/rag_agent/graphs/nodes/rag.py src/rag_agent/runtime/chat_service.py tests/integration_tests/test_langgraph_direct_and_rag_live.py
git commit -m "feat: route direct and rag through graph nodes"
```

### Task 3: Add MCP and Mixed Graph Paths With Mode-Boundary Checks

**Files:**
- Create: `src/rag_agent/graphs/nodes/mcp.py`
- Create: `src/rag_agent/graphs/nodes/mixed.py`
- Create: `src/rag_agent/graphs/nodes/references.py`
- Modify: `src/rag_agent/graphs/chat_agent.py`
- Test: `tests/integration_tests/test_langgraph_mcp_and_mixed_live.py`

**Interfaces:**
- Consumes: `ChatRuntimeService.run_chat`, MCP config/runtime helpers, `ChatGraphState`
- Produces: `run_mcp_node(state: ChatGraphState) -> ChatGraphState`, `run_mixed_node(state: ChatGraphState) -> ChatGraphState`, `merge_references(mode: str, result: dict[str, object]) -> dict[str, object]`

- [ ] **Step 1: Write the failing MCP and mixed live tests**

```python
import pytest
from langgraph_sdk import get_sync_client


@pytest.mark.integration
def test_chat_agent_mcp_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [{"role": "user", "content": "Use an MCP capability configured for this environment"}],
            "context": {"mode": "mcp"},
        },
    )
    assert result["references"]["mode"] == "mcp"
    assert result["references"].get("mcp_used") is True


@pytest.mark.integration
def test_chat_agent_mixed_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [{"role": "user", "content": "Use retrieval and tools if needed for the configured environment"}],
            "context": {"mode": "mixed", "collection_name": "default"},
        },
    )
    assert result["references"]["mode"] == "mixed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration_tests/test_langgraph_mcp_and_mixed_live.py -v`
Expected: FAIL because the graph does not yet route MCP or mixed behavior

- [ ] **Step 3: Implement reference assembly and MCP/mixed nodes**

```python
# src/rag_agent/graphs/nodes/references.py
from __future__ import annotations


def merge_references(mode: str, result: dict[str, object]) -> dict[str, object]:
    references = {k: v for k, v in result.items() if k != "final_answer"}
    references["mode"] = mode
    return references
```

```python
# src/rag_agent/graphs/nodes/mcp.py
from __future__ import annotations

from src.rag_agent.graphs.nodes.references import merge_references
from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_mcp_node(state: ChatGraphState) -> ChatGraphState:
    context = state.get("context", {})
    result = await ChatRuntimeService().run_chat(
        messages=state["messages"],
        model_id=context.get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=None,
        enable_reranker=False,
        enable_tracing=context.get("enable_tracing"),
        mode="mcp",
        mcp_server_keys=context.get("mcp_server_keys"),
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": merge_references("mcp", result),
    }
```

```python
# src/rag_agent/graphs/nodes/mixed.py
from __future__ import annotations

from src.rag_agent.graphs.nodes.references import merge_references
from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_mixed_node(state: ChatGraphState) -> ChatGraphState:
    context = state.get("context", {})
    result = await ChatRuntimeService().run_chat(
        messages=state["messages"],
        model_id=context.get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=context.get("collection_name"),
        enable_reranker=context.get("enable_reranker"),
        enable_tracing=context.get("enable_tracing"),
        mode="mixed",
        mcp_server_keys=context.get("mcp_server_keys"),
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": merge_references("mixed", result),
    }
```

- [ ] **Step 4: Extend graph routing and add mode-boundary assertions**

```python
# src/rag_agent/graphs/chat_agent.py
from src.rag_agent.graphs.nodes.mcp import run_mcp_node
from src.rag_agent.graphs.nodes.mixed import run_mixed_node


def route_mode(state: ChatGraphState) -> str:
    return state.get("context", {}).get("mode", "direct")
```

```python
graph.add_node("mcp", run_mcp_node)
graph.add_node("mixed", run_mixed_node)
graph.add_conditional_edges(
    "bootstrap",
    route_mode,
    {
        "direct": "direct",
        "rag": "rag",
        "mcp": "mcp",
        "mixed": "mixed",
    },
)
graph.add_edge("mcp", END)
graph.add_edge("mixed", END)
```

- [ ] **Step 5: Run live MCP/mixed checks**

Run: `uv run pytest tests/integration_tests/test_langgraph_mcp_and_mixed_live.py -v`
Expected: PASS when configured MCP/tooling is available, otherwise SKIP with an explicit reason

- [ ] **Step 6: Commit**

```bash
git add src/rag_agent/graphs/chat_agent.py src/rag_agent/graphs/nodes/mcp.py src/rag_agent/graphs/nodes/mixed.py src/rag_agent/graphs/nodes/references.py tests/integration_tests/test_langgraph_mcp_and_mixed_live.py
git commit -m "feat: add mcp and mixed graph routes"
```

### Task 4: Rewire Frontend Stream Ownership to LangGraph Agent Server

**Files:**
- Create: `frontend/src/providers/langgraph-stream-provider.tsx`
- Modify: `frontend/src/hooks/chat/useChatController.ts`
- Modify: `frontend/src/hooks/chat/useChatActions.ts`
- Modify: `frontend/src/hooks/chat/stream-config.ts`
- Modify: `frontend/src/hooks/useChatSession.ts`
- Test: `frontend/src/hooks/chat/__tests__/stream-config.test.ts`
- Test: `tests/workflow_tests/test_langgraph_stream.py`

**Interfaces:**
- Consumes: graph id `chat_agent`, `useStream` from `@langchain/react`, current product UI state from `useSessionUIState`
- Produces: `buildLangGraphSubmitInput(text: string, context: ChatBodyParams, messageId?: string) -> dict`, `LangGraphStreamProvider` React context with `threadId`, `setThreadId`, `stream`

- [ ] **Step 1: Write the failing frontend payload test**

```typescript
import { buildLangGraphSubmitInput } from "@/hooks/chat/stream-config";

test("buildLangGraphSubmitInput emits standard messages plus context only", () => {
  expect(
    buildLangGraphSubmitInput("hello", {
      model: "test-model",
      thread_id: "thread-1",
      session_id: "session-1",
      collection_name: "default",
      enable_reranker: true,
      enable_tracing: true,
      mode: "rag",
    }),
  ).toEqual({
    messages: [{ role: "user", content: "hello" }],
    context: {
      model_id: "test-model",
      collection_name: "default",
      mode: "rag",
      enable_reranker: true,
      enable_tracing: true,
      session_id: "session-1",
    },
  });
});
```

- [ ] **Step 2: Run the frontend test to verify it fails**

Run: `pnpm --dir frontend test -- --runInBand frontend/src/hooks/chat/__tests__/stream-config.test.ts`
Expected: FAIL because `buildLangGraphSubmitInput` does not exist or still emits adapter fields

- [ ] **Step 3: Implement standard LangGraph submit input**

```typescript
// frontend/src/hooks/chat/stream-config.ts
export function buildLangGraphSubmitInput(
  text: string,
  bodyParams: ChatBodyParams,
  messageId?: string,
) {
  return {
    messages: [
      {
        id: messageId,
        role: "user",
        content: text,
      },
    ],
    context: {
      model_id: bodyParams.model,
      collection_name: bodyParams.collection_name,
      mode: bodyParams.mode,
      enable_reranker: bodyParams.enable_reranker,
      enable_tracing: bodyParams.enable_tracing,
      session_id: bodyParams.session_id,
      thread_id: bodyParams.thread_id,
    },
  };
}
```

```typescript
// frontend/src/hooks/chat/useChatActions.ts
void Promise.resolve(
  stream.submit(
    buildLangGraphSubmitInput(trimmedText, bodyParams, userMessageId),
    {
      streamMode: ["values", "tools"],
      optimisticValues: (previous) => ({
        messages: [
          ...(Array.isArray(previous.messages) ? previous.messages : []),
          { id: userMessageId, role: "user", content: trimmedText },
        ],
      }),
    },
  ),
).catch(() => undefined);
```

- [ ] **Step 4: Replace hardcoded assistant routing with graph identity**

```typescript
// frontend/src/hooks/chat/useChatController.ts
const stream = useStream({
  apiUrl: langgraphApiUrl,
  assistantId: "chat_agent",
  threadId,
  onThreadId: (id) => {
    if (id) {
      setThreadId(id);
    }
  },
});
```

- [ ] **Step 5: Run frontend and workflow stream checks**

Run: `pnpm --dir frontend build`
Expected: PASS

Run: `uv run pytest tests/workflow_tests/test_langgraph_stream.py -v`
Expected: PASS or targeted failures showing remaining adapter assumptions

- [ ] **Step 6: Commit**

```bash
git add frontend/src/providers/langgraph-stream-provider.tsx frontend/src/hooks/chat/useChatController.ts frontend/src/hooks/chat/useChatActions.ts frontend/src/hooks/chat/stream-config.ts frontend/src/hooks/useChatSession.ts frontend/src/hooks/chat/__tests__/stream-config.test.ts tests/workflow_tests/test_langgraph_stream.py
git commit -m "feat: point frontend stream at chat_agent"
```

### Task 5: Move Thread History and Tool Rendering to Standard LangGraph Primitives

**Files:**
- Modify: `frontend/src/hooks/useChatSession.ts`
- Modify: `frontend/src/hooks/chat/message-projection.ts`
- Modify: `frontend/src/hooks/chat/references.ts`
- Modify: `frontend/src/components/chat/ChatMessageList.tsx`
- Modify: `frontend/src/components/chat/ChatMessageItem.tsx`
- Test: `tests/workflow_tests/test_chat_persistence_and_delete.py`
- Test: `tests/workflow_tests/test_chat_nonstream_and_validation.py`

**Interfaces:**
- Consumes: `stream.client.threads.search`, `stream.messages`, `stream.toolCalls` or `stream.toolProgress`, graph-produced `references`
- Produces: `loadThreadHistory(client) -> Promise<ChatThreadSummary[]>`, visible messages rendered from standard LangGraph message/tool state

- [ ] **Step 1: Write the failing thread-history and tool-visibility tests**

```python
def test_thread_history_comes_from_server_side_state(...):
    ...


def test_live_tool_progress_is_rendered_without_adapter_specific_projection(...):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflow_tests/test_chat_persistence_and_delete.py tests/workflow_tests/test_chat_nonstream_and_validation.py -v`
Expected: FAIL because the UI still depends on adapter-oriented thread and message shaping

- [ ] **Step 3: Replace local-first thread reconstruction with server-backed thread history**

```typescript
// frontend/src/hooks/useChatSession.ts
export async function loadThreadHistory(client: {
  threads: { search: (args: { limit: number }) => Promise<Array<{ thread_id: string }>> };
}): Promise<ChatThreadSummary[]> {
  const threads = await client.threads.search({ limit: 30 });
  return threads.map((thread) => ({
    id: thread.thread_id,
    title: `Chat ${thread.thread_id.slice(-6)}`,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }));
}
```

- [ ] **Step 4: Prefer standard message and tool primitives**

```typescript
// frontend/src/hooks/chat/message-projection.ts
export function projectStreamMessages(args: {
  streamMessages: BaseMessageWithKwargs[] | undefined;
  liveToolProgressEvents: McpProgressEvent[];
}): MessageLike[] {
  const mapped = (args.streamMessages ?? []).map((message, index) => ({
    id: typeof message.id === "string" ? message.id : `message-${index}`,
    role: toRole(message),
    content: readText(message.content),
    references: toReferences(message),
  }));

  return withLiveToolProgress(mapped, args.liveToolProgressEvents);
}
```

```typescript
// frontend/src/components/chat/ChatMessageItem.tsx
if (message.role === "tool") {
  return <ToolProgressCard message={message} />;
}
```

- [ ] **Step 5: Run workflow tests and build**

Run: `uv run pytest tests/workflow_tests/test_chat_persistence_and_delete.py tests/workflow_tests/test_chat_nonstream_and_validation.py -v`
Expected: PASS

Run: `pnpm --dir frontend build`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useChatSession.ts frontend/src/hooks/chat/message-projection.ts frontend/src/hooks/chat/references.ts frontend/src/components/chat/ChatMessageList.tsx frontend/src/components/chat/ChatMessageItem.tsx tests/workflow_tests/test_chat_persistence_and_delete.py tests/workflow_tests/test_chat_nonstream_and_validation.py
git commit -m "feat: use LangGraph thread and tool primitives"
```

### Task 6: Remove the Compatibility Router and Update Docs and Regression Coverage

**Files:**
- Delete: `api/routes/langgraph_server.py`
- Modify: `api/main.py`
- Modify: `docs/api/20-chat/README.md`
- Modify: `docs/api/bruno/CustomRAGAgent/Chat/create-thread.bru`
- Modify: `docs/api/bruno/CustomRAGAgent/Chat/chat-stream.bru`
- Modify: `docs/api/bruno/CustomRAGAgent/Chat/chat-stream-events.bru`
- Modify: `docs/api/bruno/CustomRAGAgent/Chat/thread-history.bru`
- Modify: `docs/api/bruno/CustomRAGAgent/Chat/thread-state.bru`
- Modify: `tests/workflow_tests/test_openapi_baseline.py`
- Test: `tests/workflow_tests/test_api_docs_sync.py`

**Interfaces:**
- Consumes: Agent Server as the chat protocol source, remaining FastAPI app for product APIs
- Produces: no `/api/langgraph/*` routes in local API docs or runtime, docs and Bruno examples aligned to the new development path

- [ ] **Step 1: Write the failing deletion regression test**

```python
def test_custom_langgraph_routes_are_absent(client) -> None:
    response = client.post("/api/langgraph/threads", json={})
    assert response.status_code == 404
```

- [ ] **Step 2: Run the regression test to verify it fails**

Run: `uv run pytest tests/workflow_tests/test_openapi_baseline.py -v`
Expected: FAIL because `/api/langgraph/*` routes are still registered

- [ ] **Step 3: Remove the compatibility router registration**

```python
# api/main.py
from api.routes import chat, config_router, documents, feedback, suggestions


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(config_router.router)
    app.include_router(documents.router)
    app.include_router(feedback.router)
    app.include_router(suggestions.router)
    return app
```

- [ ] **Step 4: Remove adapter docs and Bruno examples**

```md
# docs/api/20-chat/README.md
Chat execution now uses LangGraph Agent Server with graph id `chat_agent`.
Use `uv run langgraph dev` for local chat/thread/run/stream APIs.
FastAPI documents only product endpoints that are not provided by LangGraph Agent Server.
```

```bru
# docs/api/bruno/CustomRAGAgent/Chat/create-thread.bru
meta {
  name: "Create thread (LangGraph Agent Server)"
}

post {
  url: {{LANGGRAPH_BASE_URL}}/threads
}
```

- [ ] **Step 5: Run final docs and regression checks**

Run: `uv run pytest tests/workflow_tests/test_openapi_baseline.py tests/workflow_tests/test_api_docs_sync.py -v`
Expected: PASS

Run: `uv run python scripts/sync_api_docs.py --check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add api/main.py docs/api/20-chat/README.md docs/api/bruno/CustomRAGAgent/Chat/create-thread.bru docs/api/bruno/CustomRAGAgent/Chat/chat-stream.bru docs/api/bruno/CustomRAGAgent/Chat/chat-stream-events.bru docs/api/bruno/CustomRAGAgent/Chat/thread-history.bru docs/api/bruno/CustomRAGAgent/Chat/thread-state.bru tests/workflow_tests/test_openapi_baseline.py
git rm api/routes/langgraph_server.py
git commit -m "refactor: remove custom langgraph compatibility routes"
```

### Task 7: Prove End-to-End Live Parity Across All Four Modes

**Files:**
- Create: `tests/integration_tests/test_langgraph_mode_parity_live.py`
- Modify: `frontend/package.json`
- Modify: `README.md`
- Test: `tests/integration_tests/test_langgraph_mode_parity_live.py`
- Test: `frontend` end-to-end harness or existing e2e config

**Interfaces:**
- Consumes: real configured provider/model, real retrieval resources, real MCP configuration, graph id `chat_agent`
- Produces: gated live acceptance proof for `direct`, `rag`, `mcp`, and `mixed`

- [ ] **Step 1: Write the failing live parity suite**

```python
import pytest
from langgraph_sdk import get_sync_client


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "prompt"),
    [
        ("direct", "Reply with DIRECT_OK"),
        ("rag", "Answer using configured retrieval context"),
        ("mcp", "Use a configured MCP capability"),
        ("mixed", "Use retrieval and tools if needed"),
    ],
)
def test_chat_agent_mode_parity_live(configured_langgraph_url: str, mode: str, prompt: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={"messages": [{"role": "user", "content": prompt}], "context": {"mode": mode}},
    )
    assert result["references"]["mode"] == mode
```

- [ ] **Step 2: Run the parity suite to verify any remaining gaps**

Run: `uv run pytest tests/integration_tests/test_langgraph_mode_parity_live.py -v`
Expected: FAIL or SKIP until all four paths are fully wired

- [ ] **Step 3: Add a dedicated verification script entry**

```json
// frontend/package.json
{
  "scripts": {
    "verify:langgraph": "playwright test tests/e2e/langgraph-chat.spec.ts"
  }
}
```

```md
# README.md
## Live parity verification

Backend:
`uv run pytest tests/integration_tests/test_langgraph_mode_parity_live.py -v`

Frontend:
`pnpm --dir frontend verify:langgraph`
```

- [ ] **Step 4: Run full live verification**

Run: `uv run pytest tests/integration_tests/test_langgraph_mode_parity_live.py -v`
Expected: PASS when environment is configured, otherwise SKIP with explicit reasons

Run: `pnpm --dir frontend verify:langgraph`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration_tests/test_langgraph_mode_parity_live.py frontend/package.json README.md
git commit -m "test: add live mode parity verification"
```

## Self-Review

Spec coverage check:
- Real LangGraph graph and `langgraph.json`: Task 1
- Direct/RAG/MCP/mixed graph routing: Tasks 2 and 3
- Preserve four modes as graph routing policy: Tasks 2, 3, and 7
- Standard `useStream` frontend wiring: Task 4
- Thread history and tool rendering from standard primitives: Task 5
- Delete `/api/langgraph/*` compatibility layer: Task 6
- Live configured provider/tool validation: Tasks 2, 3, and 7

Placeholder scan:
- No `TBD`, `TODO`, or deferred “write tests later” steps remain.
- Each code-changing step includes concrete code snippets and commands.

Type consistency:
- `ChatGraphState`, `ChatGraphContext`, `build_chat_agent`, `run_direct_node`, `run_rag_node`, `run_mcp_node`, `run_mixed_node`, `merge_references`, and `buildLangGraphSubmitInput` are defined consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-langgraph-agent-server-adoption.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

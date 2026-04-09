"""Tests for DraftAnswer and RAG/MCP flow.

Flow (see README): one path per turn — Router sends to search/follow-up interpretation (RAG), select_mcp (MCP), or direct.
DraftAnswer picks final_answer by mode; it does not merge RAG + MCP in one answer.
"""

import asyncio
import importlib
from typing import cast
from unittest.mock import patch

from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.agent_state import State
from src.rag_agent.langgraph.nodes.answer_generator import AnswerFromDocs
from src.rag_agent.langgraph.nodes.answer_generator import DraftAnswer
from src.rag_agent.langgraph.nodes.mixed import CallMCPTools


def test_route_after_answer_from_docs_mixed_falls_back_when_citations_are_ungrounded():
    graph_mod = importlib.import_module("src.rag_agent.langgraph.graph")
    answer_node = AnswerFromDocs()

    state: State = {
        "user_request": "Solve the following equation: x^2 - 5x + 6 = 0 use tools",
        "messages": [],
        "mode": "mixed",
        "mcp_tool_match": True,
        "reranker_docs": [
            {
                "page_content": "Install OpenCode CLI on macOS.",
                "metadata": {"source": "opencode.md", "page": 1},
            }
        ],
        "citations": [{"source": "opencode.md", "page": 1}],
    }

    def fake_try_structured_path(
        llm: object,
        the_question: str,
        context: str,
        chat_history_text: str,
        num_sources: int,
        run_config: RunnableConfig,
    ) -> str | None:
        _ = (llm, the_question, context, chat_history_text, num_sources, run_config)
        return "The solutions are x = 2 and x = 3 [1]."

    with patch.object(answer_node, "_try_structured_path", side_effect=fake_try_structured_path), patch(
        "src.rag_agent.langgraph.nodes.answer_generator.get_llm",
        return_value=object(),
    ):
        answer_state = answer_node.invoke(state)

    assert answer_state["rag_has_citations"] is False

    combined_state = {**state, **answer_state}

    route = graph_mod.route_after_answer_from_docs(combined_state)

    assert route == "select_mcp"


def test_draft_answer_rag_not_substantive_mcp_substantive():
    """MCP path (mode=mcp): final is mcp_answer only; RAG is not merged."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "mcp",
        "rag_answer": "I'm ready to help with your questions about the provided context.",
        "mcp_answer": "The derivative of sin(x^2) with respect to x is 2x cos(x^2).",
        "mcp_used": True,
        "mcp_tools_used": ["differentiate"],
    }
    result = draft.invoke(state)
    final_answer = cast(str, result["final_answer"])
    assert final_answer == "The derivative of sin(x^2) with respect to x is 2x cos(x^2)."
    assert result["mcp_used"] is True
    assert "I'm ready to help" not in final_answer


def test_draft_answer_rag_path():
    """RAG path (mode=rag): final is rag_answer only; stale mcp_answer from checkpoint ignored."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "rag",
        "route": "search",
        "rag_answer": "OCI CLI can be installed via pip [1]. See the docs for details.",
        "mcp_answer": "Run: pip install oci-cli",  # stale from previous turn
        "citations": [{"source": "doc1", "page": "1"}],
    }
    result = draft.invoke(state)
    final_answer = cast(str, result["final_answer"])
    assert "OCI CLI can be installed" in final_answer
    assert "pip install oci-cli" not in final_answer
    assert result["mcp_used"] is False
    assert result["mcp_tools_used"] == []


def test_draft_answer_mcp_path():
    """MCP path (mode=mcp): final is mcp_answer only."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "mcp",
        "route": "select_mcp",
        "mcp_answer": "Result: 42",
        "mcp_used": True,
        "mcp_tools_used": ["calculate"],
    }
    result = draft.invoke(state)
    final_answer = cast(str, result["final_answer"])
    assert final_answer == "Result: 42"
    assert result.get("mcp_used") is True
    assert result.get("mcp_tools_used") == ["calculate"]


def test_draft_answer_mixed_tools_used():
    """Mixed path with tools_used: prefer mcp_answer."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "mixed",
        "rag_answer": "From docs [1].",
        "mcp_answer": "Run: pip install oci-cli",
        "mcp_tools_used": ["install_cli"],
    }
    result = draft.invoke(state)
    final_answer = cast(str, result["final_answer"])
    assert final_answer == "Run: pip install oci-cli"


def test_draft_answer_mixed_rag_wins_preserves_citations():
    """Mixed path should keep citations when the chosen final answer is the RAG answer."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "mixed",
        "rag_answer": "From docs [1].",
        "mcp_answer": "",
        "mcp_used": True,
        "mcp_tools_used": ["docs.search"],
        "citations": [{"source": "doc1", "page": "1"}],
    }

    result = draft.invoke(state)

    assert result["final_answer"] == "From docs [1]."
    assert result["citations"] == [{"source": "doc1", "page": "1"}]
    assert result["mcp_used"] is True
    assert result["mcp_tools_used"] == ["docs.search"]


def test_draft_answer_direct():
    """Direct path (mode=direct): final is direct_answer only."""
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "direct",
        "route": "direct",
        "direct_answer": "Hello, how can I help?",
    }
    result = draft.invoke(state)
    final_answer = cast(str, result["final_answer"])
    assert final_answer == "Hello, how can I help?"


def test_call_mcp_tools_mixed_uses_selected_tools(monkeypatch):
    class StubTool:
        def __init__(self, name: str) -> None:
            self.name = name

    captured: dict[str, object] = {}

    def fake_get_mcp_tools(*, server_keys=None, run_config=None):
        captured["server_keys"] = server_keys
        captured["run_config"] = run_config
        return [StubTool("math.solve"), StubTool("docs.search")]

    def fake_invoke_get_mcp_answer(
        self,
        question,
        messages,
        model_id,
        server_keys,
        tools,
        require_tool_call,
        run_config,
    ):
        _ = self
        captured["question"] = question
        captured["messages"] = list(messages)
        captured["model_id"] = model_id
        captured["forwarded_server_keys"] = server_keys
        captured["tools"] = [tool.name for tool in (tools or [])]
        captured["require_tool_call"] = require_tool_call
        captured["forwarded_run_config"] = run_config
        return "tool answer", ["math.solve"]

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.mixed.get_mcp_tools", fake_get_mcp_tools)
    monkeypatch.setattr(CallMCPTools, "_invoke_get_mcp_answer", fake_invoke_get_mcp_answer)

    node = CallMCPTools()
    state = cast(
        State,
        cast(
            object,
            {
                "user_request": "Solve this",
                "messages": [],
                "mode": "mixed",
                "selected_mcp_tool_names": ["math.solve"],
                "selected_mcp_tool_descriptions": ["Solve math expressions"],
            },
        ),
    )
    config = cast(
        RunnableConfig,
        cast(object, {"configurable": {"model_id": "test-model", "mcp_server_keys": ["math"]}}),
    )

    result = node.invoke(state, config=config)

    assert captured["tools"] == ["math.solve"]
    assert captured["server_keys"] == ["math"]
    assert captured["require_tool_call"] is True
    assert result["mcp_answer"] == "tool answer"
    assert result["mcp_tools_used"] == ["math.solve"]
    assert result["mcp_used"] is True
    assert result["latest_answer"] == "tool answer"




def _make_state(user_request: str = "question", mode: str = "mixed") -> dict[str, object]:
    return {
        "user_request": user_request,
        "messages": [],
        "mode": mode,
    }


def test_call_mcp_tools_mixed_ainvoke_uses_selected_tools(monkeypatch):
    class StubTool:
        def __init__(self, name: str) -> None:
            self.name = name

    captured: dict[str, object] = {}

    async def fake_get_mcp_tools_async(*, server_keys=None, run_config=None):
        captured["server_keys"] = server_keys
        captured["run_config"] = run_config
        return [StubTool("math.solve"), StubTool("docs.search")]

    async def fake_get_mcp_answer_async(
        question,
        chat_history,
        model_id=None,
        server_keys=None,
        tools=None,
        require_tool_call=False,
        run_config=None,
    ):
        captured["question"] = question
        captured["chat_history"] = list(chat_history)
        captured["model_id"] = model_id
        captured["forwarded_server_keys"] = server_keys
        captured["tools"] = [tool.name for tool in (tools or [])]
        captured["require_tool_call"] = require_tool_call
        captured["forwarded_run_config"] = run_config
        return "async tool answer", ["math.solve"]

    monkeypatch.setattr(
        "src.rag_agent.langgraph.nodes.mixed.get_mcp_tools_async", fake_get_mcp_tools_async
    )

    node = CallMCPTools()
    state = cast(
        State,
        cast(
            object,
            {
                "user_request": "Solve this async",
                "messages": [],
                "mode": "mixed",
                "selected_mcp_tool_names": ["math.solve"],
                "selected_mcp_tool_descriptions": ["Solve math expressions"],
            },
        ),
    )
    config = cast(
        RunnableConfig,
        cast(object, {"configurable": {"model_id": "test-model", "mcp_server_keys": ["math"]}}),
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_answer_async",
        new=fake_get_mcp_answer_async,
    ):
        result = asyncio.run(node.ainvoke(state, config=config))

    assert captured["tools"] == ["math.solve"]
    assert captured["server_keys"] == ["math"]
    assert captured["require_tool_call"] is True
    assert result["mcp_answer"] == "async tool answer"
    assert result["mcp_tools_used"] == ["math.solve"]
    assert result["mcp_used"] is True


def test_call_mcp_tools_mixed_without_selected_tools_does_not_require_tool_call(monkeypatch):
    module = importlib.import_module("src.rag_agent.langgraph.nodes.mixed")
    call_node = module.CallMCPTools()
    captured: dict[str, object] = {}

    def fake_invoke(
        question: str,
        messages: list[object],
        model_id: str | None,
        server_keys: list[str] | None,
        tools: list[object] | None = None,
        require_tool_call: bool = False,
        run_config: dict[str, object] | None = None,
    ) -> tuple[str, list[str]]:
        captured["question"] = question
        captured["messages"] = messages
        captured["model_id"] = model_id
        captured["server_keys"] = server_keys
        captured["tools"] = tools
        captured["require_tool_call"] = require_tool_call
        captured["run_config"] = run_config
        return ("fallback to direct answer", [])

    monkeypatch.setattr(call_node, "_invoke_get_mcp_answer", fake_invoke)

    state = _make_state(user_request="what tools do you have access to?", mode="mixed")
    state["selected_mcp_tool_names"] = []
    config = {"configurable": {"mode": "mixed", "model_id": "test-model"}}

    out = call_node.invoke(state, config=config)

    assert captured["require_tool_call"] is False
    assert out["mcp_answer"] == "fallback to direct answer"
    assert out["mcp_tools_used"] == []
    assert out["mcp_used"] is False


def test_call_mcp_tools_mixed_ainvoke_without_selected_tools_does_not_require_tool_call(
    monkeypatch,
):
    module = importlib.import_module("src.rag_agent.langgraph.nodes.mixed")
    call_node = module.CallMCPTools()
    captured: dict[str, object] = {}

    async def fake_get_mcp_answer_async(
        question: str,
        messages: list[object],
        model_id: str | None = None,
        server_keys: list[str] | None = None,
        tools: list[object] | None = None,
        require_tool_call: bool = False,
        run_config: dict[str, object] | None = None,
    ) -> tuple[str, list[str]]:
        captured["question"] = question
        captured["messages"] = messages
        captured["model_id"] = model_id
        captured["server_keys"] = server_keys
        captured["tools"] = tools
        captured["require_tool_call"] = require_tool_call
        captured["run_config"] = run_config
        return ("fallback to direct answer", [])

    monkeypatch.setattr(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    state = _make_state(user_request="what tools do you have access to?", mode="mixed")
    state["selected_mcp_tool_names"] = []
    config = {"configurable": {"mode": "mixed", "model_id": "test-model"}}

    out = asyncio.run(call_node.ainvoke(state, config=config))

    assert captured["require_tool_call"] is False
    assert out["mcp_answer"] == "fallback to direct answer"
    assert out["mcp_tools_used"] == []
    assert out["mcp_used"] is False


def test_draft_answer_mixed_mcp_wins_clears_stale_citations():
    draft = DraftAnswer()
    state: State = {
        "user_request": "test",
        "messages": [],
        "mode": "mixed",
        "rag_answer": "From docs [1].",
        "mcp_answer": "Tool result",
        "mcp_used": True,
        "mcp_tools_used": ["calculator.solve_equation"],
        "citations": [{"source": "doc1", "page": "1"}],
    }

    result = draft.invoke(state)

    assert result["final_answer"] == "Tool result"
    assert result["citations"] == []
    assert result["mcp_used"] is True
    assert result["mcp_tools_used"] == ["calculator.solve_equation"]

import importlib
from typing import cast

from src.rag_agent.agent_state import State


def _make_state(user_request="question", messages=None) -> State:
    return {
        "user_request": user_request,
        "messages": messages or [],
    }


def test_router_direct_mode_routes_direct():
    router_mod = importlib.import_module("rag_agent.langgraph.nodes.router")
    router_cls = getattr(router_mod, "Router")

    router = router_cls()
    state = _make_state("hello")
    config = {"configurable": {"mode": "direct"}}

    out = router.invoke(state, config=config)

    assert out["route"] == "direct"
    assert out["mode"] == "direct"


def test_router_mcp_mode_routes_select_mcp_and_clears_rag_fields():
    router_mod = importlib.import_module("rag_agent.langgraph.nodes.router")
    router_cls = getattr(router_mod, "Router")

    router = router_cls()
    state = _make_state("use tools")
    config = {"configurable": {"mode": "mcp"}}

    out = router.invoke(state, config=config)

    assert out["route"] == "select_mcp"
    assert out["mode"] == "mcp"
    # RAG-related fields should be cleared to avoid mixed execution in this turn
    assert out.get("rag_answer") is None
    assert out.get("retriever_docs") == []
    assert out.get("reranker_docs") == []
    assert out.get("citations") == []


def test_router_mixed_routes_to_mcp_when_selector_matches_tools(monkeypatch):
    router_mod = importlib.import_module("rag_agent.langgraph.nodes.router")
    router_cls = getattr(router_mod, "Router")

    router = router_cls()
    state = _make_state("tool question")
    config = {"configurable": {"mode": "mixed", "mcp_server_keys": ["math"]}}

    async def fake_selector(*args, **kwargs):
        _ = args
        _ = kwargs
        return {
            "question": "tool question",
            "limit": 5,
            "selected_tools": [
                {
                    "canonical_name": "math.solve",
                    "tool_name": "solve",
                    "server_key": "math",
                    "description": "Solve math expressions",
                    "input_schema": {},
                }
            ],
            "total_tools": 1,
            "selection_failed": False,
            "failure": None,
        }

    monkeypatch.setattr(router_mod, "select_mcp_tools_for_question_async", fake_selector)

    out = router.invoke(state, config=config)

    assert out["route"] == "search"
    assert out["mode"] == "mixed"
    assert out.get("mcp_tool_match") is True
    assert out.get("selected_mcp_tool_names") == ["math.solve"]
    assert out.get("selected_mcp_tool_descriptions") == ["Solve math expressions"]
    assert out.get("mcp_tools_used") in (None, [])


def test_router_mixed_routes_to_rag_when_no_tools_match(monkeypatch):
    router_mod = importlib.import_module("rag_agent.langgraph.nodes.router")
    router_cls = getattr(router_mod, "Router")

    router = router_cls()
    state = _make_state("knowledge question")
    config = {"configurable": {"mode": "mixed", "mcp_server_keys": ["default"]}}

    async def fake_selector(*args, **kwargs):
        _ = args
        _ = kwargs
        return {
            "question": "knowledge question",
            "limit": 5,
            "selected_tools": [],
            "total_tools": 0,
            "selection_failed": False,
            "failure": None,
        }

    monkeypatch.setattr(router_mod, "select_mcp_tools_for_question_async", fake_selector)

    out = router.invoke(state, config=config)

    assert out["route"] == "search"
    assert out["mode"] == "mixed"
    assert out.get("mcp_tool_match") is False
    assert out.get("selected_mcp_tool_names") == []
    assert out.get("selected_mcp_tool_descriptions") == []
    assert out.get("mcp_tools_used") in (None, [])


def test_router_rag_mode_routes_search():
    router_mod = importlib.import_module("rag_agent.langgraph.nodes.router")
    router_cls = getattr(router_mod, "Router")

    router = router_cls()
    state = _make_state("kb question")
    config = {"configurable": {"mode": "rag"}}

    out = router.invoke(state, config=config)

    assert out["route"] == "search"
    assert out["mode"] == "rag"


def test_route_after_followup_interpreter_routes_mcp_followup_to_select_mcp():
    graph_mod = importlib.import_module("src.rag_agent.langgraph.graph")

    route = graph_mod.route_after_followup_interpreter(
        cast(State, cast(object, {"followup_intent": "mcp_followup"}))
    )

    assert route == "select_mcp"

from __future__ import annotations

from langchain_core.documents import Document

from src.rag_agent.graphs import mcp_policies as mod


def test_workflow_policy_for_request_activates_only_for_matching_mode_and_terms(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "MCP_WORKFLOW_POLICY": {
                    "enabled": True,
                    "apply_modes": ["mixed"],
                    "activation_terms": ["invoice"],
                    "required_capabilities": ["classify", "extract"],
                    "tool_capability_map": {"tool_a": ["classify"], "tool_b": ["extract"]},
                }
            },
        )(),
    )

    active = mod.workflow_policy_for_request(
        mode="mixed",
        question="Process this invoice end to end.",
    )
    inactive = mod.workflow_policy_for_request(
        mode="mcp",
        question="Process this invoice end to end.",
    )

    assert active is not None
    assert active["required_capabilities"] == ["classify", "extract"]
    assert inactive is None


def test_enforce_workflow_policy_reports_missing_capabilities() -> None:
    applied, missing, message = mod.enforce_workflow_policy(
        policy={
            "required_capabilities": ["classify", "extract", "create"],
            "tool_capability_map": {
                "tool_a": ["classify"],
                "tool_b": ["extract"],
            },
        },
        tools_used=["tool_a"],
        tool_invocations=[{"tool_name": "tool_b", "result": "ok"}],
    )

    assert applied is True
    assert missing == ["create"]
    assert message is not None
    assert "missing required steps: create" in message.lower()


def test_oracle_retrieval_helpers_use_tool_state_and_tool_calls() -> None:
    retrieval_state = {
        "docs": [Document(page_content="Net 30", metadata={"source": "northway.md"})],
        "error": "database unavailable",
    }
    tool_invocations = [{"tool_name": "oracle_retrieval", "result": "Net 30"}]

    assert mod.oracle_retrieval_used_without_context(
        retrieval_state={"docs": []},
        retrieval_docs=[],
        tools_used=["oracle_retrieval"],
        tool_invocations=tool_invocations,
    )
    assert (
        mod.oracle_retrieval_error(
            retrieval_state=retrieval_state,
            tools_used=["oracle_retrieval"],
            tool_invocations=tool_invocations,
        )
        == "database unavailable"
    )


def test_mixed_tool_supplemental_context_ignores_retrieval_tool() -> None:
    context = mod.mixed_tool_supplemental_context(
        [
            {"tool_name": "oracle_retrieval", "result": "doc"},
            {"tool_name": "calculator", "args": {"expression": "5+5"}, "result": "10"},
        ]
    )

    assert context == "Tool: calculator\nArgs: {'expression': '5+5'}\nResult: 10"


def test_runtime_flags_and_trivial_answer_use_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "REQUIRE_TOOL_CALL": True,
                "MCP_REPEATED_WORKFLOW_CONTROLLER": True,
                "ENABLE_PERSISTENT_MEMORY": True,
                "LANGGRAPH_SQLITE_PATH": "/tmp/langgraph.sqlite",
            },
        )(),
    )

    assert mod.require_tool_call_enabled() is True
    assert mod.repeated_workflow_controller_enabled() is True
    assert mod.workflow_checkpoint_path() == "/tmp/langgraph.sqlite"
    assert mod.is_trivial_answer(".")
    assert not mod.is_trivial_answer("real answer")

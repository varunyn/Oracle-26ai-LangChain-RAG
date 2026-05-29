from __future__ import annotations

import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, tool

from src.rag_agent.workflows.mcp_repeated import _cached_tool, run_repeated_mcp_workflow
from src.rag_agent.workflows.work_unit_extraction import extract_work_units_from_tool_invocations


def test_extract_work_units_from_generic_multi_item_tool_result() -> None:
    invocations = [
        {
            "tool_name": "list_things",
            "args": {},
            "result": '[{"type": "text", "text": "{\\"items\\": [{\\"id\\": \\"a\\"}, {\\"id\\": \\"b\\"}]}"}]',
        }
    ]

    units = extract_work_units_from_tool_invocations(invocations)

    assert [unit["id"] for unit in units] == ["a", "b"]
    assert [unit["payload"] for unit in units] == [{"id": "a"}, {"id": "b"}]


def test_extract_work_units_from_arbitrary_named_top_level_list_tool_result() -> None:
    invocations = [
        {
            "tool_name": "list_documents",
            "args": {},
            "result": (
                '[{"type": "text", "text": "{\\"fileList\\": ['
                '{\\"fileName\\": \\"a.pdf\\"}, {\\"fileName\\": \\"b.pdf\\"}]}"}]'
            ),
        }
    ]

    units = extract_work_units_from_tool_invocations(invocations)

    assert [unit["payload"] for unit in units] == [{"fileName": "a.pdf"}, {"fileName": "b.pdf"}]


def test_extract_work_units_from_arbitrary_named_nested_envelope_tool_result() -> None:
    invocations = [
        {
            "tool_name": "list_documents",
            "args": {},
            "result": (
                '{"response": {"totallyCustomQueueName": ['
                '{"fileName": "a.pdf"}, {"fileName": "b.pdf"}]}}'
            ),
        }
    ]

    units = extract_work_units_from_tool_invocations(invocations)

    assert [unit["payload"] for unit in units] == [{"fileName": "a.pdf"}, {"fileName": "b.pdf"}]


def test_extract_work_units_ignores_nested_detail_collections() -> None:
    invocations = [
        {
            "tool_name": "inspect_thing",
            "args": {},
            "result": '{"record": {"id": "parent", "details": [{"id": "a"}, {"id": "b"}]}}',
        }
    ]

    assert extract_work_units_from_tool_invocations(invocations) == []


def test_cached_tool_preserves_content_and_artifact_response_format() -> None:
    calls = 0

    async def retrieve(query: str) -> tuple[str, list[dict[str, str]]]:
        """Retrieve context."""
        nonlocal calls
        calls += 1
        return ("retrieved context", [{"query": query}])

    tool_with_artifact = StructuredTool.from_function(
        coroutine=retrieve,
        name="oracle_retrieval",
        description="Retrieve context.",
        response_format="content_and_artifact",
    )
    cached = _cached_tool(tool_with_artifact, {})

    async def invoke_twice() -> tuple[object, object]:
        tool_call = {
            "type": "tool_call",
            "id": "call-1",
            "name": "oracle_retrieval",
            "args": {"query": "payment terms"},
        }
        first = await cached.ainvoke(tool_call)
        second = await cached.ainvoke(tool_call)
        return first, second

    first_result, second_result = asyncio.run(invoke_twice())

    assert calls == 1
    assert isinstance(first_result, ToolMessage)
    assert first_result.content == "retrieved context"
    assert first_result.artifact == [{"query": "payment terms"}]
    assert isinstance(second_result, ToolMessage)
    assert second_result.artifact == first_result.artifact


def test_run_repeated_mcp_workflow_discovers_processes_then_finalizes(monkeypatch) -> None:
    @tool
    def list_things() -> str:
        """List things."""
        return "ok"

    @tool
    def mutate_thing(payload: str) -> str:
        """Mutate one thing."""
        return payload

    @tool
    def email_summary(summary: str) -> str:
        """Send final summary."""
        return summary

    calls: list[tuple[str, list[str]]] = []

    async def fake_get_answer(**kwargs):
        question = str(kwargs["question"])
        tools = kwargs["tools"]
        calls.append((question, [tool.name for tool in tools]))
        if len(calls) == 1:
            return (
                "discovered",
                ["list_things"],
                [
                    {
                        "tool_name": "list_things",
                        "args": {},
                        "result": '{"items": [{"id": "a"}, {"id": "b"}]}',
                    }
                ],
            )
        if len(calls) in {2, 3}:
            return ("processed", ["mutate_thing"], [{"tool_name": "mutate_thing", "result": "ok"}])
        return ("finalized", ["email_summary"], [{"tool_name": "email_summary", "result": "ok"}])

    async def fake_select_discovery_tools(**kwargs):
        tools = kwargs["tools"]
        return [tool for tool in tools if tool.name == "list_things"]

    async def fake_select_processing_tools(**kwargs):
        tools = kwargs["tools"]
        return [tool for tool in tools if tool.name == "mutate_thing"]

    async def fake_select_finalization_tools(**kwargs):
        tools = kwargs["tools"]
        return [tool for tool in tools if tool.name == "email_summary"]

    async def fake_select_context_tools(**kwargs):
        _ = kwargs
        return []

    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_discovery_tools",
        fake_select_discovery_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_processing_tools",
        fake_select_processing_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_finalization_tools",
        fake_select_finalization_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_context_tools",
        fake_select_context_tools,
    )

    result = asyncio.run(
        run_repeated_mcp_workflow(
            question="Review the thing queue and send a summary",
            model_id=None,
            tools=[list_things, mutate_thing, email_summary],
            run_config={"configurable": {"thread_id": "thread-1"}},
            require_tool_call=True,
            get_answer=fake_get_answer,
        )
    )

    assert result is not None
    answer, tools_used, invocations = result
    assert answer == "finalized"
    assert tools_used == ["list_things", "mutate_thing", "email_summary"]
    assert calls[0][1] == ["list_things"]
    assert calls[1][1] == ["mutate_thing"]
    assert calls[2][1] == ["mutate_thing"]
    assert calls[-1][1] == ["email_summary"]
    assert "email_summary" in calls[-1][1]
    assert "Do not perform any final completion" in calls[1][0]
    assert "Do not perform any final completion" in calls[2][0]
    assert len([inv for inv in invocations if inv["tool_name"] == "mutate_thing"]) == 2


def test_run_repeated_mcp_workflow_stops_when_discovery_finds_no_queue(monkeypatch) -> None:
    @tool
    def list_things() -> str:
        """List things."""
        return "ok"

    async def fake_get_answer(**kwargs):
        _ = kwargs
        return (
            "No matching work items were found.",
            ["list_things"],
            [{"tool_name": "list_things", "args": {}, "result": '{"items": []}'}],
        )

    async def fake_select_discovery_tools(**kwargs):
        return list(kwargs["tools"])

    async def fake_select_processing_tools(**kwargs):
        return list(kwargs["tools"])

    async def fake_select_finalization_tools(**kwargs):
        return list(kwargs["tools"])

    async def fake_select_context_tools(**kwargs):
        _ = kwargs
        return []

    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_discovery_tools",
        fake_select_discovery_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_processing_tools",
        fake_select_processing_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_finalization_tools",
        fake_select_finalization_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_context_tools",
        fake_select_context_tools,
    )

    result = asyncio.run(
        run_repeated_mcp_workflow(
            question="Review the thing queue and send a summary",
            model_id=None,
            tools=[list_things],
            run_config={"configurable": {"thread_id": "thread-1"}},
            require_tool_call=True,
            get_answer=fake_get_answer,
        )
    )

    assert result is not None
    answer, tools_used, invocations = result
    assert "could not identify a work queue" in answer
    assert "No matching work items were found." in answer
    assert tools_used == ["list_things"]
    assert invocations == [{"tool_name": "list_things", "args": {}, "result": '{"items": []}'}]


def test_run_repeated_mcp_workflow_does_not_reuse_finalized_checkpoint_for_new_run(
    monkeypatch,
    tmp_path,
) -> None:
    @tool
    def list_things() -> str:
        """List things."""
        return "ok"

    @tool
    def mutate_thing(payload: str) -> str:
        """Mutate one thing."""
        return payload

    @tool
    def email_summary(summary: str) -> str:
        """Send final summary."""
        return summary

    async def fake_select_discovery_tools(**kwargs):
        return [tool for tool in kwargs["tools"] if tool.name == "list_things"]

    async def fake_select_processing_tools(**kwargs):
        return [tool for tool in kwargs["tools"] if tool.name == "mutate_thing"]

    async def fake_select_finalization_tools(**kwargs):
        return [tool for tool in kwargs["tools"] if tool.name == "email_summary"]

    async def fake_select_context_tools(**kwargs):
        _ = kwargs
        return []

    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_discovery_tools",
        fake_select_discovery_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_processing_tools",
        fake_select_processing_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_finalization_tools",
        fake_select_finalization_tools,
    )
    monkeypatch.setattr(
        "src.rag_agent.workflows.mcp_repeated.select_repeated_workflow_context_tools",
        fake_select_context_tools,
    )

    processing_calls = 0

    async def fake_get_answer(**kwargs):
        nonlocal processing_calls
        tools = kwargs["tools"]
        tool_names = {tool.name for tool in tools}
        if tool_names == {"list_things"}:
            return (
                "discovered",
                ["list_things"],
                [
                    {
                        "tool_name": "list_things",
                        "args": {},
                        "result": '{"items": [{"id": "a"}, {"id": "b"}]}',
                    }
                ],
            )
        if tool_names == {"mutate_thing"}:
            processing_calls += 1
            return ("processed", ["mutate_thing"], [{"tool_name": "mutate_thing", "result": "ok"}])
        return ("finalized", ["email_summary"], [{"tool_name": "email_summary", "result": "ok"}])

    async def run_once() -> None:
        result = await run_repeated_mcp_workflow(
            question="Review the thing queue and send a summary",
            model_id=None,
            tools=[list_things, mutate_thing, email_summary],
            run_config={"configurable": {"thread_id": "thread-1"}},
            require_tool_call=True,
            get_answer=fake_get_answer,
            checkpoint_path=tmp_path / "workflow.sqlite",
        )
        assert result is not None

    asyncio.run(run_once())
    asyncio.run(run_once())

    assert processing_calls == 4

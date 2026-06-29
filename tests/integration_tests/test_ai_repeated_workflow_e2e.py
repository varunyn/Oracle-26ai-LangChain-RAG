"""Live AI e2e checks for repeated MCP workflow control.

These tests intentionally use the configured real chat model. They keep external side effects
local by providing in-process tools instead of calling OIC or other MCP servers.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from api.settings import get_settings
from src.rag_agent.graphs.chat_agent import chat_agent


def _ai_workflow_e2e_enabled() -> bool:
    return (
        os.environ.get("RUN_INTEGRATION_TESTS") == "1"
        and os.environ.get("OCI_INTEGRATION_TESTS") == "1"
        and os.environ.get("AI_WORKFLOW_E2E_TESTS") == "1"
    )


def _ai_workflow_e2e_model_id() -> str:
    return os.environ.get("AI_WORKFLOW_E2E_MODEL_ID", "xai.grok-4.20-0309-reasoning")


@pytest.mark.integration
@pytest.mark.skipif(
    not _ai_workflow_e2e_enabled(),
    reason=(
        "Set RUN_INTEGRATION_TESTS=1 OCI_INTEGRATION_TESTS=1 "
        "AI_WORKFLOW_E2E_TESTS=1 to run live AI workflow e2e tests"
    ),
)
def test_ai_repeated_workflow_controller_processes_every_local_tool_item(monkeypatch) -> None:
    monkeypatch.setenv("MCP_REPEATED_WORKFLOW_CONTROLLER", "true")
    monkeypatch.setenv("REQUIRE_TOOL_CALL", "true")
    get_settings.cache_clear()

    completed_item_ids: list[str] = []
    summaries: list[str] = []

    @tool("list_work_items")
    def list_work_items() -> str:
        """List all work items that must be processed."""
        return json.dumps(
            {
                "items": [
                    {"id": "alpha", "description": "Alpha item"},
                    {"id": "beta", "description": "Beta item"},
                ]
            }
        )

    @tool("mark_item_done")
    def mark_item_done(itemId: str) -> str:  # noqa: N803 - mirrors common MCP JSON schemas
        """Mark one work item done by id."""
        completed_item_ids.append(itemId)
        return json.dumps({"itemId": itemId, "status": "completed"})

    @tool("send_summary")
    def send_summary(summary: str) -> str:
        """Send the final workflow summary."""
        summaries.append(summary)
        return json.dumps({"status": "sent"})

    async def fake_load_adapter_tools(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [list_work_items, mark_item_done, send_summary]

    monkeypatch.setattr(
        "src.rag_agent.runtime.mcp_turn.load_adapter_tools",
        fake_load_adapter_tools,
    )

    async def run() -> dict[str, object]:
        state = await chat_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the available tools to list all work items. For each listed item, "
                            "mark the item done by its id. Continue until all listed items are handled, "
                            "then send one final summary."
                        )
                    )
                ]
            },
            {
                "configurable": {
                    "thread_id": "ai-workflow-e2e",
                    "mode": "mcp",
                    "model_id": _ai_workflow_e2e_model_id(),
                    "mcp_server_keys": ["local-test"],
                }
            },
        )
        assistant = state["messages"][-1]
        assert isinstance(assistant, AIMessage)
        references = getattr(assistant, "additional_kwargs", {}) or {}
        return {
            "final_answer": getattr(assistant, "content", ""),
            "error": references.get("error"),
            "mcp_tools_used": references.get("mcp_tools_used") or [],
        }

    result = asyncio.run(run())

    try:
        diagnostic = json.dumps(
            {
                "result": result,
                "completed_item_ids": completed_item_ids,
                "summaries": summaries,
            },
            default=str,
            indent=2,
        )
        assert result.get("error") is None
        assert set(completed_item_ids) == {"alpha", "beta"}, diagnostic
        assert len(completed_item_ids) == 2, diagnostic
        assert len(summaries) == 1, diagnostic
        tools_used = result.get("mcp_tools_used")
        assert isinstance(tools_used, Sequence) and not isinstance(tools_used, (str, bytes))
        assert {"list_work_items", "mark_item_done", "send_summary"}.issubset(
            {str(tool_name) for tool_name in tools_used}
        ), diagnostic
    finally:
        get_settings.cache_clear()


@pytest.mark.integration
@pytest.mark.skipif(
    not _ai_workflow_e2e_enabled(),
    reason=(
        "Set RUN_INTEGRATION_TESTS=1 OCI_INTEGRATION_TESTS=1 "
        "AI_WORKFLOW_E2E_TESTS=1 to run live AI workflow e2e tests"
    ),
)
def test_ai_repeated_workflow_reuses_shared_context_for_same_vendor(monkeypatch) -> None:
    monkeypatch.setenv("MCP_REPEATED_WORKFLOW_CONTROLLER", "true")
    monkeypatch.setenv("REQUIRE_TOOL_CALL", "true")
    get_settings.cache_clear()

    context_lookup_calls: list[str] = []
    created_item_ids: list[str] = []
    summaries: list[str] = []

    @tool("list_vendor_work_items")
    def list_vendor_work_items() -> str:
        """List all vendor work items that must be processed."""
        return json.dumps(
            {
                "workBatch": [
                    {
                        "id": "invoice-a",
                        "vendorName": "Summit Technologies",
                        "amount": "100.00",
                    },
                    {
                        "id": "invoice-b",
                        "vendorName": "Summit Technologies",
                        "amount": "200.00",
                    },
                ]
            }
        )

    @tool("lookup_vendor_policy")
    def lookup_vendor_policy(vendorName: str) -> str:  # noqa: N803 - mirrors MCP schemas
        """Look up reusable processing policy for one vendor name."""
        context_lookup_calls.append(vendorName)
        return json.dumps(
            {"vendorName": vendorName, "policyId": "summit-policy", "maxTermDays": 45}
        )

    @tool("create_vendor_transaction")
    def create_vendor_transaction(
        itemId: str,  # noqa: N803 - mirrors MCP schemas
        vendorName: str,  # noqa: N803 - mirrors MCP schemas
        policyId: str,  # noqa: N803 - mirrors MCP schemas
    ) -> str:
        """Create one transaction for a listed work item using a vendor policy id."""
        created_item_ids.append(itemId)
        return json.dumps(
            {
                "itemId": itemId,
                "vendorName": vendorName,
                "policyId": policyId,
                "status": "created",
            }
        )

    @tool("send_vendor_summary")
    def send_vendor_summary(summary: str) -> str:
        """Send the final vendor workflow summary."""
        summaries.append(summary)
        return json.dumps({"status": "sent"})

    async def fake_load_adapter_tools(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [
            list_vendor_work_items,
            lookup_vendor_policy,
            create_vendor_transaction,
            send_vendor_summary,
        ]

    monkeypatch.setattr(
        "src.rag_agent.runtime.mcp_turn.load_adapter_tools",
        fake_load_adapter_tools,
    )

    async def run() -> dict[str, object]:
        state = await chat_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use tools to list all vendor work items. For each item, create one "
                            "vendor transaction. Before creating transactions, look up vendor policy "
                            "exactly once per unique vendor and reuse that policy for every item from "
                            "the same vendor. Continue until every listed item is handled, then send "
                            "one final summary."
                        )
                    )
                ]
            },
            {
                "configurable": {
                    "thread_id": "ai-workflow-shared-context-e2e",
                    "mode": "mcp",
                    "model_id": _ai_workflow_e2e_model_id(),
                    "mcp_server_keys": ["local-test"],
                }
            },
        )
        assistant = state["messages"][-1]
        assert isinstance(assistant, AIMessage)
        references = getattr(assistant, "additional_kwargs", {}) or {}
        return {
            "final_answer": getattr(assistant, "content", ""),
            "error": references.get("error"),
            "mcp_tools_used": references.get("mcp_tools_used") or [],
        }

    result = asyncio.run(run())

    try:
        diagnostic = json.dumps(
            {
                "result": result,
                "context_lookup_calls": context_lookup_calls,
                "created_item_ids": created_item_ids,
                "summaries": summaries,
            },
            default=str,
            indent=2,
        )
        assert result.get("error") is None
        assert context_lookup_calls == ["Summit Technologies"], diagnostic
        assert set(created_item_ids) == {"invoice-a", "invoice-b"}, diagnostic
        assert len(created_item_ids) == 2, diagnostic
        assert len(summaries) == 1, diagnostic
        tools_used = result.get("mcp_tools_used")
        assert isinstance(tools_used, Sequence) and not isinstance(tools_used, (str, bytes))
        assert {
            "list_vendor_work_items",
            "lookup_vendor_policy",
            "create_vendor_transaction",
            "send_vendor_summary",
        }.issubset({str(tool_name) for tool_name in tools_used}), diagnostic
    finally:
        get_settings.cache_clear()

"""MCP agent turn helpers for chat runtime modes."""

from __future__ import annotations

from langchain_core.tools import BaseTool


def question_explicitly_references_mcp_tools(
    question: str,
    mcp_tools: list[BaseTool],
) -> bool:
    lower_question = question.strip().lower()
    if not lower_question:
        return False
    for tool in mcp_tools:
        tool_name = str(getattr(tool, "name", "") or "").strip().lower()
        if not tool_name:
            continue
        if tool_name in lower_question:
            return True
        humanized = tool_name.replace("_", " ")
        if humanized in lower_question:
            return True
    return False


def tool_failure_summary(tool_invocations: list[dict[str, object]]) -> str | None:
    failed_tools: list[str] = []
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool_name") or "").strip()
        error_text = str(invocation.get("error") or "").strip()
        if error_text:
            if tool_name and tool_name not in failed_tools:
                failed_tools.append(tool_name)
    if not failed_tools:
        return None
    joined = ", ".join(failed_tools)
    return f"Workflow failed because tool execution failed: {joined}. See tool output for details."

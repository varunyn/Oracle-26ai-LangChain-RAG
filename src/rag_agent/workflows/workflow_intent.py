from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.rag_agent.infrastructure.oci_models import get_llm

logger = logging.getLogger(__name__)


class WorkflowIntentDecision(BaseModel):
    """Model decision for whether the request needs repeated work-unit control."""

    use_repeated_workflow: bool = Field(
        description=(
            "True only when the user asks to process multiple independent work units "
            "through the same steps before finalization."
        )
    )
    reason: str = Field(default="", description="Short reason for the decision.")


class WorkflowDiscoveryToolDecision(BaseModel):
    """Model decision for which tools may be used to discover the work queue."""

    tool_names: list[str] = Field(
        default_factory=list,
        description="Tool names that should be available during queue discovery only.",
    )
    reason: str = Field(default="", description="Short reason for the selection.")


async def should_use_repeated_workflow(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> bool:
    """Let the model classify repeated workflow intent without domain-specific terms."""

    if not question.strip() or not tools:
        return False

    def _invoke() -> WorkflowIntentDecision:
        llm = get_llm(model_id=model_id, temperature=0, max_tokens=256)
        structured = llm.with_structured_output(WorkflowIntentDecision)
        result = structured.invoke(_classification_messages(question=question, tools=tools), config=run_config)
        if isinstance(result, WorkflowIntentDecision):
            return result
        if isinstance(result, dict):
            return WorkflowIntentDecision.model_validate(result)
        raw_result = llm.invoke(
            _json_classification_messages(question=question, tools=tools),
            config=run_config,
        )
        return _parse_json_decision(raw_result)

    try:
        decision = await asyncio.to_thread(_invoke)
    except Exception as exc:  # noqa: BLE001
        logger.info("repeated_workflow_intent_classifier_failed error=%s", exc)
        return False
    return bool(decision.use_repeated_workflow)


async def select_repeated_workflow_discovery_tools(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> list[BaseTool]:
    """Let the model restrict discovery to queue-establishing tools."""

    if not question.strip() or not tools:
        return list(tools)

    def _invoke() -> WorkflowDiscoveryToolDecision:
        llm = get_llm(model_id=model_id, temperature=0, max_tokens=512)
        structured = llm.with_structured_output(WorkflowDiscoveryToolDecision)
        result = structured.invoke(
            _discovery_tool_messages(question=question, tools=tools),
            config=run_config,
        )
        if isinstance(result, WorkflowDiscoveryToolDecision):
            return result
        if isinstance(result, dict):
            return WorkflowDiscoveryToolDecision.model_validate(result)
        raw_result = llm.invoke(
            _json_discovery_tool_messages(question=question, tools=tools),
            config=run_config,
        )
        return _parse_json_tool_decision(raw_result)

    try:
        decision = await asyncio.to_thread(_invoke)
    except Exception as exc:  # noqa: BLE001
        logger.info("repeated_workflow_discovery_tool_selector_failed error=%s", exc)
        return list(tools)

    selected_names = {name.strip() for name in decision.tool_names if name.strip()}
    selected = [tool for tool in tools if str(getattr(tool, "name", "") or "").strip() in selected_names]
    return selected or list(tools)


async def select_repeated_workflow_processing_tools(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> list[BaseTool]:
    """Let the model restrict per-unit processing to non-finalization tools."""

    return await _select_repeated_workflow_tools(
        question=question,
        tools=tools,
        model_id=model_id,
        run_config=run_config,
        system_prompt=(
            "Select the tools that may be used while processing exactly one known work unit "
            "inside a repeated workflow. Include tools that inspect, validate, retrieve reusable "
            "context, or create/update the one current unit. Exclude tools that list or discover "
            "the full queue, and exclude tools that send final summaries, notifications, emails, "
            "or completion reports after all units are done. Return only exact tool names from "
            "the available tool list."
        ),
    )


async def select_repeated_workflow_finalization_tools(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> list[BaseTool]:
    """Let the model restrict finalization to reporting/notification tools."""

    return await _select_repeated_workflow_tools(
        question=question,
        tools=tools,
        model_id=model_id,
        run_config=run_config,
        system_prompt=(
            "Select the tools that may be used only after every work unit in a repeated workflow "
            "has reached a terminal state. Include tools that send final summaries, notifications, "
            "emails, completion reports, or other final reporting. Exclude tools that list the queue "
            "or process/create/update individual work units. Return only exact tool names from the "
            "available tool list."
        ),
    )


async def select_repeated_workflow_context_tools(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None = None,
) -> list[BaseTool]:
    """Let the model select read-only context tools safe to cache across work units."""

    return await _select_repeated_workflow_tools(
        question=question,
        tools=tools,
        model_id=model_id,
        run_config=run_config,
        system_prompt=(
            "Select read-only tools whose results may be reused across multiple work units "
            "when called with the same arguments in a repeated workflow. Include tools that "
            "retrieve reference data, policies, database facts, vendor/customer context, or "
            "other lookup information. Exclude tools that list the work queue, classify or "
            "extract one unit, create records, update records, send notifications, or finalize "
            "the workflow. Return only exact tool names from the available tool list."
        ),
    )


async def _select_repeated_workflow_tools(
    *,
    question: str,
    tools: Sequence[BaseTool],
    model_id: str | None,
    run_config: RunnableConfig | None,
    system_prompt: str,
) -> list[BaseTool]:
    if not question.strip() or not tools:
        return list(tools)

    def _invoke() -> WorkflowDiscoveryToolDecision:
        llm = get_llm(model_id=model_id, temperature=0, max_tokens=512)
        structured = llm.with_structured_output(WorkflowDiscoveryToolDecision)
        result = structured.invoke(
            _tool_selection_messages(question=question, tools=tools, system_prompt=system_prompt),
            config=run_config,
        )
        if isinstance(result, WorkflowDiscoveryToolDecision):
            return result
        if isinstance(result, dict):
            return WorkflowDiscoveryToolDecision.model_validate(result)
        raw_result = llm.invoke(
            _json_tool_selection_messages(question=question, tools=tools, system_prompt=system_prompt),
            config=run_config,
        )
        return _parse_json_tool_decision(raw_result)

    try:
        decision = await asyncio.to_thread(_invoke)
    except Exception as exc:  # noqa: BLE001
        logger.info("repeated_workflow_tool_selector_failed error=%s", exc)
        return list(tools)

    selected_names = {name.strip() for name in decision.tool_names if name.strip()}
    selected = [tool for tool in tools if str(getattr(tool, "name", "") or "").strip() in selected_names]
    return selected or list(tools)


def _classification_messages(*, question: str, tools: Sequence[BaseTool]) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(
            content=(
                "Decide whether this user request needs a repeated workflow controller.\n"
                "Use repeated workflow only when all are true:\n"
                "1. The request requires a set or queue of multiple independent work units.\n"
                "2. The same process should be run separately for each work unit.\n"
                "3. A final summary, notification, or completion step should happen after all units.\n"
                "Return false for normal single-answer questions, one-off tool calls, retrieval, "
                "or requests where a tool result may contain nested lists that are not the work queue."
            )
        ),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _tool_selection_messages(
    *,
    question: str,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _json_tool_selection_messages(
    *,
    question: str,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(
            content=(
                f"{system_prompt}\nRespond with JSON only using this exact shape: "
                '{"tool_names": ["tool_name"], "reason": "short reason"}.'
            )
        ),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _discovery_tool_messages(*, question: str, tools: Sequence[BaseTool]) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(
            content=(
                "Select the tools that may be used only to establish the complete work queue "
                "for a repeated workflow. Include tools that list, search, enumerate, or fetch "
                "the set of independent work units. Exclude tools that process one unit, mutate "
                "external state, create records, update records, send notifications, produce final "
                "summaries, or validate details after a unit is already known. Return only exact "
                "tool names from the available tool list."
            )
        ),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _json_discovery_tool_messages(*, question: str, tools: Sequence[BaseTool]) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(
            content=(
                "Select queue-discovery tools for a repeated workflow. Respond with JSON only "
                'using this exact shape: {"tool_names": ["tool_name"], "reason": "short reason"}. '
                "Include only tools that establish the complete list/set of work units. Exclude "
                "tools that process, create, update, notify, summarize, or finalize."
            )
        ),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _json_classification_messages(*, question: str, tools: Sequence[BaseTool]) -> list[object]:
    tool_summary = "\n".join(_tool_line(tool) for tool in tools[:40])
    return [
        SystemMessage(
            content=(
                "Classify whether the request needs a repeated workflow controller. "
                "Respond with JSON only using this exact shape: "
                '{"use_repeated_workflow": true|false, "reason": "short reason"}.\n'
                "Set use_repeated_workflow true only when the user asks to obtain or use "
                "multiple independent work units, process each one separately, and then "
                "perform final reporting or notification after all units are terminal."
            )
        ),
        HumanMessage(content=f"Available tools:\n{tool_summary}\n\nUser request:\n{question}"),
    ]


def _parse_json_decision(raw_result: object) -> WorkflowIntentDecision:
    text = _result_text(raw_result).strip()
    if not text:
        return WorkflowIntentDecision(use_repeated_workflow=False, reason="Empty classifier result")
    try:
        return WorkflowIntentDecision.model_validate_json(text)
    except Exception:  # noqa: BLE001
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return WorkflowIntentDecision.model_validate(parsed)
        except Exception:  # noqa: BLE001
            pass
    return WorkflowIntentDecision(use_repeated_workflow=False, reason="Unparseable classifier result")


def _parse_json_tool_decision(raw_result: object) -> WorkflowDiscoveryToolDecision:
    text = _result_text(raw_result).strip()
    if not text:
        return WorkflowDiscoveryToolDecision(reason="Empty selector result")
    try:
        return WorkflowDiscoveryToolDecision.model_validate_json(text)
    except Exception:  # noqa: BLE001
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return WorkflowDiscoveryToolDecision.model_validate(parsed)
        except Exception:  # noqa: BLE001
            pass
    return WorkflowDiscoveryToolDecision(reason="Unparseable selector result")


def _result_text(raw_result: object) -> str:
    content = getattr(raw_result, "content", raw_result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _tool_line(tool: BaseTool) -> str:
    name = str(getattr(tool, "name", "") or "").strip()
    description = str(getattr(tool, "description", "") or "").strip().replace("\n", " ")
    if len(description) > 240:
        description = f"{description[:237]}..."
    return f"- {name}: {description}" if description else f"- {name}"

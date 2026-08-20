from __future__ import annotations

from typing import cast

from api.settings import get_settings
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidence

ORACLE_RETRIEVAL_TOOL_NAME = "oracle_retrieval"
NO_ORACLE_CONTEXT_ANSWER = "I don't know the answer from the selected Oracle collection."
ORACLE_RETRIEVAL_FAILED_ANSWER = (
    "I couldn't retrieve context from the selected Oracle collection because retrieval failed. "
    "Please try again after the database is available."
)


def _to_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def workflow_policy_for_request(*, mode: str, question: str) -> dict[str, object] | None:
    policy_raw = getattr(get_settings(), "MCP_WORKFLOW_POLICY", {})
    if not isinstance(policy_raw, dict) or not policy_raw:
        return None
    enabled = bool(policy_raw.get("enabled", True))
    if not enabled:
        return None
    apply_modes = _to_string_list(policy_raw.get("apply_modes")) or ["mixed"]
    if mode not in {m.lower() for m in apply_modes}:
        return None
    activation_terms = [
        term.lower() for term in _to_string_list(policy_raw.get("activation_terms"))
    ]
    lower_question = question.strip().lower()
    if activation_terms and not any(term in lower_question for term in activation_terms):
        return None
    required_capabilities = _to_string_list(policy_raw.get("required_capabilities"))
    tool_capability_map_raw = policy_raw.get("tool_capability_map")
    if not required_capabilities or not isinstance(tool_capability_map_raw, dict):
        return None
    tool_capability_map: dict[str, list[str]] = {}
    for tool_name, capabilities in tool_capability_map_raw.items():
        normalized_tool_name = str(tool_name).strip().lower()
        if not normalized_tool_name:
            continue
        caps = _to_string_list(capabilities)
        if caps:
            tool_capability_map[normalized_tool_name] = [cap.lower() for cap in caps]
    if not tool_capability_map:
        return None
    return {
        "required_capabilities": [cap.lower() for cap in required_capabilities],
        "tool_capability_map": tool_capability_map,
        "failure_message": str(policy_raw.get("failure_message") or "").strip(),
    }


def require_tool_call_enabled() -> bool:
    return bool(getattr(get_settings(), "REQUIRE_TOOL_CALL", False))


def repeated_workflow_controller_enabled() -> bool:
    return bool(getattr(get_settings(), "MCP_REPEATED_WORKFLOW_CONTROLLER", False))


def workflow_checkpoint_path() -> str | None:
    settings = get_settings()
    raw_path = str(getattr(settings, "LANGGRAPH_SQLITE_PATH", "") or "").strip()
    return raw_path or None


def enforce_workflow_policy(
    *,
    policy: dict[str, object] | None,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> tuple[bool, list[str], str | None]:
    if policy is None:
        return False, [], None
    required_capabilities = _to_string_list(policy.get("required_capabilities"))
    tool_capability_map = cast(dict[str, list[str]], policy.get("tool_capability_map") or {})
    if not required_capabilities or not tool_capability_map:
        return False, [], None
    called_capabilities: set[str] = set()
    for tool_name in _called_tool_names(
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    ):
        for capability in tool_capability_map.get(tool_name, []):
            called_capabilities.add(capability.lower())
    missing = [cap for cap in required_capabilities if cap.lower() not in called_capabilities]
    if not missing:
        return True, [], None
    default_message = (
        "Workflow validation failed. Missing required steps: "
        + ", ".join(missing)
        + ". Please continue with the required workflow tools."
    )
    failure_message = str(policy.get("failure_message") or "").strip() or default_message
    return True, missing, failure_message


def is_trivial_answer(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    return not any(ch.isalnum() for ch in stripped)


def _called_tool_names(
    *,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> set[str]:
    names = {str(name).strip().lower() for name in tools_used if str(name).strip()}
    names.update(
        str(invocation.get("tool_name") or "").strip().lower()
        for invocation in tool_invocations
        if isinstance(invocation, dict) and str(invocation.get("tool_name") or "").strip()
    )
    return names


def _tool_was_called(
    *,
    tool_name: str,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> bool:
    expected = tool_name.strip().lower()
    return expected in _called_tool_names(
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    )


def oracle_retrieval_used_without_context(
    *,
    retrieval_evidence: OracleRetrievalEvidence | None,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> bool:
    if retrieval_evidence is None or retrieval_evidence.documents:
        return False
    if retrieval_evidence.error:
        return False
    return _evidence_matches_oracle_invocation(
        retrieval_evidence=retrieval_evidence,
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    )


def oracle_retrieval_error(
    *,
    retrieval_evidence: OracleRetrievalEvidence | None,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> str | None:
    if retrieval_evidence is None:
        return None
    error = str(retrieval_evidence.error or "").strip()
    if not error:
        return None
    if not _evidence_matches_oracle_invocation(
        retrieval_evidence=retrieval_evidence,
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    ):
        return None
    return error


def _evidence_matches_oracle_invocation(
    *,
    retrieval_evidence: OracleRetrievalEvidence,
    tools_used: list[str],
    tool_invocations: list[dict[str, object]],
) -> bool:
    if not _tool_was_called(
        tool_name=ORACLE_RETRIEVAL_TOOL_NAME,
        tools_used=tools_used,
        tool_invocations=tool_invocations,
    ):
        return False
    return any(
        invocation.get("tool_name") == ORACLE_RETRIEVAL_TOOL_NAME
        and invocation.get("invocation_id") == retrieval_evidence.invocation_id
        for invocation in tool_invocations
    )


def mixed_tool_supplemental_context(
    tool_invocations: list[dict[str, object]],
) -> str | None:
    blocks: list[str] = []
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool_name") or "").strip()
        if not tool_name or tool_name == ORACLE_RETRIEVAL_TOOL_NAME:
            continue
        error = str(invocation.get("error") or "").strip()
        result = str(invocation.get("result") or "").strip()
        if error:
            result = f"Error: {error}"
        if not result:
            continue
        args = invocation.get("args")
        args_text = f"\nArgs: {args}" if args not in (None, {}, []) else ""
        blocks.append(f"Tool: {tool_name}{args_text}\nResult: {result}")
    return "\n\n".join(blocks) or None

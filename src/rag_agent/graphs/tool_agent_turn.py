from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Literal, TypedDict, TypeVar, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.constants import CONFIG_KEY_CHECKPOINTER
from langgraph.runtime import Runtime

from api.settings import get_settings
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
from src.rag_agent.infrastructure.mcp_agent_executor import _build_tool_summary
from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.prompts.mcp_agent_prompts import (
    SYSTEM_PROMPT_MIXED,
    TOOL_SUMMARY_PLACEHOLDER,
)
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    latest_user_message,
    latest_user_message_id,
)
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore
from src.rag_agent.runtime.tool_agent_recipe_store import (
    DEFAULT_LEASE_DURATION_MS,
    LeaseStatus,
    LeaseToken,
    RecipeConflictError,
    StaleLeaseError,
    ToolAgentTurnRecipe,
    ToolAgentTurnRecipeStore,
)

_Result = TypeVar("_Result")

_EMPTY_MCP_SELECTION = "__tool_agent_turn_empty_mcp_selection__"
_CLAIM_RETRY_ATTEMPTS = 4
_CLAIM_RETRY_BASE_SECONDS = 0.05
_MCP_REDACTED = "<redacted>"
_MCP_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
_MCP_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_MCP_SENSITIVE_HEADER_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


class ToolAgentTurn(TypedDict):
    chat_history: list[BaseMessage]
    model_id: str
    question: str
    run_config: RunnableConfig
    system_prompt: str
    tools: list[object]
    oracle_retrieval_evidence: OracleRetrievalEvidenceStore | None
    tool_round_limit: int
    enable_reranker: bool
    lease: LeaseToken


class IncompatibleMCPConfigurationError(RuntimeError):
    """The immutable recipe no longer matches configured MCP definitions."""


def _recipe_store(parent_config: RunnableConfig) -> ToolAgentTurnRecipeStore:
    configurable = parent_config.get("configurable") or {}
    checkpointer = configurable.get(CONFIG_KEY_CHECKPOINTER)
    store = getattr(checkpointer, "recipe_store", None)
    if not isinstance(store, ToolAgentTurnRecipeStore):
        raise RuntimeError("Tool-agent modes require the saver-owned recipe store.")
    return store


def _run_id(parent_config: RunnableConfig, runtime: Runtime[ChatGraphContext]) -> str | None:
    value = getattr(runtime.execution_info, "run_id", None)
    if value is None:
        value = (parent_config.get("configurable") or {}).get("run_id")
    return str(value) if value is not None else None


def _mcp_definitions() -> dict[str, dict[str, object]]:
    from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config

    return cast(dict[str, dict[str, object]], get_mcp_servers_config())


def _mcp_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _MCP_SENSITIVE_KEYS or normalized.endswith(_MCP_SENSITIVE_KEY_SUFFIXES)


def _mcp_sensitive_header(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return any(part in normalized for part in _MCP_SENSITIVE_HEADER_PARTS)


def _mcp_redact_args(values: Sequence[object]) -> list[object]:
    redacted: list[object] = []
    redact_next = False
    for value in values:
        if redact_next:
            redacted.append(_MCP_REDACTED)
            redact_next = False
            continue
        if isinstance(value, str):
            option, separator, _argument = value.partition("=")
            if separator and _mcp_sensitive_key(option):
                redacted.append(f"{option}={_MCP_REDACTED}")
                continue
            if value.startswith("-") and _mcp_sensitive_key(value):
                redacted.append(value)
                redact_next = True
                continue
        redacted.append(_mcp_redact_value(value))
    return redacted


def _mcp_redact_value(value: object, *, key: object = "") -> object:
    if _mcp_sensitive_key(key):
        return _MCP_REDACTED
    if isinstance(value, Mapping):
        if str(key).strip().lower() in {"header", "headers"}:
            return {
                str(header): (
                    _MCP_REDACTED
                    if _mcp_sensitive_header(header)
                    else _mcp_redact_value(header_value, key=header)
                )
                for header, header_value in value.items()
            }
        return {
            str(nested_key): _mcp_redact_value(nested_value, key=nested_key)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if str(key).strip().lower() == "args":
            return _mcp_redact_args(value)
        return [_mcp_redact_value(item) for item in value]
    if callable(value):
        module = cast(str, getattr(value, "__module__", type(value).__module__))
        qualname = cast(str, getattr(value, "__qualname__", type(value).__qualname__))
        return f"{module}.{qualname}"
    return value


def _mcp_behavioral_projection(
    definitions: Mapping[str, Mapping[str, object]], keys: tuple[str, ...]
) -> dict[str, object]:
    return {key: _mcp_redact_value(definitions.get(key), key=key) for key in keys}


def _mcp_digest(keys: tuple[str, ...], _context: ChatGraphContext) -> str | None:
    if not keys:
        return None
    definitions = _mcp_definitions()
    selected = _mcp_behavioral_projection(definitions, keys)
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _resolve_mcp_selection(context: ChatGraphContext) -> tuple[tuple[str, ...], str | None]:
    definitions = _mcp_definitions()
    has_explicit_selection = "mcp_server_keys" in context
    requested = tuple(str(key) for key in (context.get("mcp_server_keys") or []))
    keys = requested if has_explicit_selection else tuple(sorted(definitions))
    missing = tuple(key for key in keys if key not in definitions)
    if missing:
        raise IncompatibleMCPConfigurationError(
            "Configured MCP servers are missing: " + ", ".join(missing)
        )
    if not keys:
        return (), None
    selected = _mcp_behavioral_projection(definitions, keys)
    digest = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return keys, digest


def _recipe_from_request(
    *,
    state: ChatGraphState,
    parent_config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
    mode: Literal["mcp", "mixed"],
) -> ToolAgentTurnRecipe:
    context = get_runtime_context(runtime)
    messages = state.get("messages", [])
    turn_id = latest_user_message_id(messages)
    thread_id = get_thread_id(runtime)
    if not thread_id or not turn_id:
        raise RuntimeError("Tool-agent modes require a durable thread_id and HumanMessage.id.")
    model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    keys, mcp_config_digest = _resolve_mcp_selection(context)
    return ToolAgentTurnRecipe(
        thread_id=thread_id,
        turn_id=turn_id,
        origin_run_id=_run_id(parent_config, runtime),
        request_id=cast(str | None, context.get("request_id")),
        session_id=cast(str | None, context.get("session_id")),
        mode=mode,
        model_key=model_id,
        collection_key=cast(str | None, context.get("collection_name")),
        mcp_server_keys=keys,
        mcp_config_digest=mcp_config_digest,
        enable_tracing=bool(context.get("enable_tracing")),
        tool_round_limit=(
            int(context["max_rounds"])
            if isinstance(context.get("max_rounds"), int)
            and not isinstance(context.get("max_rounds"), bool)
            and int(context["max_rounds"]) > 0
            else max(1, int(get_settings().MCP_MAX_ROUNDS))
        ),
        enable_reranker=(
            bool(context.get("enable_reranker"))
            if isinstance(context.get("enable_reranker"), bool)
            else False
        ),
    )


async def _load_or_create_recipe(*, state, parent_config, runtime, mode, create: bool):
    store = _recipe_store(parent_config)
    thread_id = get_thread_id(runtime)
    turn_id = latest_user_message_id(state.get("messages", []))
    if not thread_id or not turn_id:
        raise RuntimeError("Tool-agent modes require a durable thread_id and HumanMessage.id.")
    if create:
        requested = _recipe_from_request(
            state=state, parent_config=parent_config, runtime=runtime, mode=mode
        )
        try:
            result = await store.create_or_load(requested)
            recipe = result.recipe
        except RecipeConflictError:
            # Agent Server may replay setup under a new run id before its first
            # checkpoint is durable.  Run identity is provenance, not recipe
            # behavior, so preserve the original canonical origin run while
            # linking the new run below.  Any real recipe drift still fails.
            existing = await store.load((thread_id, turn_id))
            if (
                existing is None
                or replace(existing, origin_run_id=None).canonical_json()
                != replace(requested, origin_run_id=None).canonical_json()
            ):
                raise
            recipe = existing
    else:
        from src.rag_agent.runtime.tool_agent_recipe_store import MissingRecipeError

        loaded_recipe = await store.load((thread_id, turn_id))
        if loaded_recipe is None:
            raise MissingRecipeError(f"Missing recipe: {thread_id}/{turn_id}")
        recipe = loaded_recipe
    run_id = _run_id(parent_config, runtime)
    if run_id:
        await store.record_run_link((thread_id, turn_id), run_id)
    definitions = _mcp_definitions()
    current_digest = _mcp_digest(recipe.mcp_server_keys, get_runtime_context(runtime))
    if (
        any(key not in definitions for key in recipe.mcp_server_keys)
        or recipe.mcp_config_digest != current_digest
    ):
        raise IncompatibleMCPConfigurationError(
            f"Stored MCP configuration for {recipe.thread_id}/{recipe.turn_id} is incompatible."
        )
    return store, recipe


async def reconstruct_tool_agent_turn(
    *,
    state: ChatGraphState,
    parent_config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
    mode: Literal["mcp", "mixed"],
    create_recipe: bool = False,
    extra_tools: Sequence[object] = (),
    oracle_retrieval_evidence: OracleRetrievalEvidenceStore | None = None,
) -> ToolAgentTurn:
    store, recipe = await _load_or_create_recipe(
        state=state,
        parent_config=parent_config,
        runtime=runtime,
        mode=mode,
        create=create_recipe,
    )
    claim = None
    for attempt in range(_CLAIM_RETRY_ATTEMPTS):
        claim = await store.claim((recipe.thread_id, recipe.turn_id), uuid.uuid4().hex)
        if getattr(claim, "status", None) is not LeaseStatus.ALREADY_ACTIVE:
            break
        if attempt + 1 < _CLAIM_RETRY_ATTEMPTS:
            await asyncio.sleep(_CLAIM_RETRY_BASE_SECONDS * (2**attempt))
    assert claim is not None
    if claim.lease is None:
        raise RuntimeError(f"Tool-agent recipe lease unavailable: {claim.status}")
    try:
        messages = state.get("messages", [])
        effective_mode = recipe.mode
        if effective_mode == "mixed" and oracle_retrieval_evidence is None:
            oracle_retrieval_evidence = OracleRetrievalEvidenceStore.from_persisted_messages(
                messages, collection_name=recipe.collection_key or "RAG_KNOWLEDGE_BASE"
            )
        run_config = build_run_config(
            parent_config=parent_config,
            thread_id=recipe.thread_id,
            mode=recipe.mode,
            model_id=recipe.model_key,
            session_id=recipe.session_id,
            enable_tracing=recipe.enable_tracing,
            mcp_server_keys=list(recipe.mcp_server_keys),
            request_id=recipe.request_id,
        )
        mixed_tools = list(extra_tools)
        if effective_mode == "mixed" and not mixed_tools:
            from src.rag_agent.runtime import rag_runtime

            evidence = oracle_retrieval_evidence or OracleRetrievalEvidenceStore()
            retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
                collection_name=recipe.collection_key,
                filter_docs=rag_runtime.filter_retrieved_docs,
                evidence=evidence,
            )
            if retrieval_tool is not None:
                mixed_tools.append(retrieval_tool)
        adapter_server_keys = list(recipe.mcp_server_keys)
        if not adapter_server_keys and recipe.mcp_config_digest is None:
            adapter_server_keys = [_EMPTY_MCP_SELECTION]

        async def load_mcp_tools() -> list[object]:
            return cast(
                list[object],
                await load_adapter_tools(server_keys=adapter_server_keys, run_config=run_config),
            )

        mcp_tools = await run_with_lease_heartbeat(
            parent_config,
            cast(ToolAgentTurn, {"lease": claim.lease}),
            load_mcp_tools,
            runtime=runtime,
        )
        tools = [*mixed_tools, *mcp_tools]
        return {
            "chat_history": chat_history_before_latest_user(messages),
            "model_id": recipe.model_key,
            "question": latest_user_message(messages),
            "run_config": run_config,
            "system_prompt": build_tool_agent_system_prompt(tools),
            "tools": tools,
            "oracle_retrieval_evidence": oracle_retrieval_evidence,
            "tool_round_limit": recipe.tool_round_limit,
            "enable_reranker": recipe.enable_reranker,
            "lease": claim.lease,
        }
    except BaseException:
        try:
            await store.release(claim.lease)
        except StaleLeaseError:
            pass
        raise


async def release_tool_agent_turn(parent_config: RunnableConfig, turn: ToolAgentTurn) -> None:
    await _recipe_store(parent_config).release(turn["lease"])


async def release_tool_agent_turn_after_failure(
    parent_config: RunnableConfig, turn: ToolAgentTurn
) -> None:
    """Release best-effort without replacing the operation failure after a takeover."""

    try:
        await release_tool_agent_turn(parent_config, turn)
    except StaleLeaseError:
        return


async def renew_tool_agent_turn(parent_config: RunnableConfig, turn: ToolAgentTurn) -> None:
    await _recipe_store(parent_config).renew(turn["lease"])


async def mark_tool_agent_turn_terminal(
    parent_config: RunnableConfig, turn: ToolAgentTurn, terminal_message_id: str
) -> None:
    """Fence the active turn once more, then persist its terminal message id."""

    store = _recipe_store(parent_config)
    await store.renew(turn["lease"])
    await store.mark_terminal(turn["lease"], terminal_message_id)


def _lease_heartbeat_interval_seconds() -> float:
    """Renew before half of the durable lease interval elapses."""

    return float(DEFAULT_LEASE_DURATION_MS) / 2_000


async def run_with_lease_heartbeat(
    parent_config: RunnableConfig,
    turn: ToolAgentTurn,
    operation_factory: Callable[[], Awaitable[_Result]],
    *,
    runtime: Runtime[ChatGraphContext] | None = None,
) -> _Result:
    """Run one external operation while renewal failure cancels it and fails closed."""

    await renew_tool_agent_turn(parent_config, turn)
    _refresh_runtime_heartbeat(runtime)
    operation_task = asyncio.ensure_future(operation_factory())

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(_lease_heartbeat_interval_seconds())
            await renew_tool_agent_turn(parent_config, turn)
            _refresh_runtime_heartbeat(runtime)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait(
            {operation_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat_task in done:
            await heartbeat_task
            raise RuntimeError("Lease heartbeat stopped unexpectedly")
        result = operation_task.result()
        await renew_tool_agent_turn(parent_config, turn)
        _refresh_runtime_heartbeat(runtime)
        return result
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        if not operation_task.done():
            operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)


def _refresh_runtime_heartbeat(runtime: Runtime[ChatGraphContext] | None) -> None:
    heartbeat = getattr(runtime, "heartbeat", None)
    if callable(heartbeat):
        heartbeat()


def build_tool_agent_system_prompt(tools: Sequence[object]) -> str:
    return cast(
        str,
        SYSTEM_PROMPT_MIXED.replace(
            TOOL_SUMMARY_PLACEHOLDER,
            cast(str, _build_tool_summary(cast(Sequence, tools))),
        ),
    )


async def prepare_tool_agent_turn(
    *,
    state: ChatGraphState,
    parent_config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
    mode: Literal["mcp", "mixed"],
    extra_tools: Sequence[object] = (),
    oracle_retrieval_evidence: OracleRetrievalEvidenceStore | None = None,
) -> ToolAgentTurn:
    return await reconstruct_tool_agent_turn(
        state=state,
        parent_config=parent_config,
        runtime=runtime,
        mode=mode,
        create_recipe=True,
        extra_tools=extra_tools,
        oracle_retrieval_evidence=oracle_retrieval_evidence,
    )


__all__ = [
    "ToolAgentTurn",
    "build_tool_agent_system_prompt",
    "IncompatibleMCPConfigurationError",
    "prepare_tool_agent_turn",
    "reconstruct_tool_agent_turn",
    "release_tool_agent_turn",
    "release_tool_agent_turn_after_failure",
    "renew_tool_agent_turn",
    "mark_tool_agent_turn_terminal",
    "run_with_lease_heartbeat",
]

"""App-side MCP client wiring around LangChain's first-party MCP adapter.

``MCPAdapter`` delegates MCP sessions, transports, and LangChain tool wrapping to
FastMCP 4. This module maps UI-managed or seed MCP server config + per-run
``RunnableConfig`` to a FastMCP ``ClientGroup``, caches adapters/tools, and must
not reimplement MCP wire protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypedDict, cast

from fastmcp.client.group import ClientGroup
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool, StructuredTool

from .mcp_oauth import build_oauth_headers_supplier
from .mcp_settings import get_mcp_servers_config, get_mcp_settings

logger = logging.getLogger(__name__)

_client_lock = asyncio.Lock()
_client_cache: dict[str, MCPAdapter] = {}
_tool_cache: dict[str, list[BaseTool]] = {}


class AdapterConnectionConfig(TypedDict, total=False):
    transport: str
    url: str
    headers: dict[str, Any]
    command: str
    args: list[str]
    env: dict[str, str]
    auth: Any
    timeout: Any
    cwd: str
    keep_alive: bool


def _extract_configurable(run_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(run_config, Mapping):
        return {}
    configurable = run_config.get("configurable")
    if isinstance(configurable, Mapping):
        return cast(Mapping[str, Any], configurable)
    return {}


def _extract_server_keys_from_run_config(run_config: Mapping[str, Any] | None) -> list[str] | None:
    configurable = _extract_configurable(run_config)
    selected = configurable.get("mcp_server_keys")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        return None
    keys = [str(item).strip() for item in selected if str(item).strip()]
    return keys or None


def _extract_config_override(
    run_config: Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    configurable = _extract_configurable(run_config)
    override = configurable.get("mcp_servers_config_override")
    if not isinstance(override, Mapping):
        return None
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_key, raw_value in override.items():
        key = str(raw_key).strip()
        if key and isinstance(raw_value, Mapping):
            normalized[key] = cast(Mapping[str, Any], raw_value)
    return normalized


def _select_server_keys(
    configured_servers: Mapping[str, Mapping[str, Any]],
    *,
    server_keys: Sequence[str] | None,
    run_config: Mapping[str, Any] | None,
) -> list[str]:
    configured_keys = list(configured_servers.keys())
    if not configured_keys:
        return []

    requested_keys = [key.strip() for key in server_keys or [] if key.strip()]
    if not requested_keys:
        requested_from_run_config = _extract_server_keys_from_run_config(run_config)
        if requested_from_run_config:
            requested_keys = requested_from_run_config

    if requested_keys:
        return [key for key in requested_keys if key in configured_servers]

    return configured_keys


def _coerce_headers(raw_headers: object) -> dict[str, Any]:
    if not isinstance(raw_headers, Mapping):
        return {}
    return {
        str(key): value
        for key, value in raw_headers.items()
        if str(key).strip() and value is not None
    }


async def _resolve_server_auth_headers(server_config: Mapping[str, Any]) -> dict[str, str]:
    raw_auth = server_config.get("auth")
    if not isinstance(raw_auth, Mapping):
        return {}

    auth_type = str(raw_auth.get("type") or "none").strip().lower()
    if auth_type == "bearer":
        token = str(raw_auth.get("bearer_token") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    if auth_type == "oauth_client_credentials":
        try:
            refresh_skew_seconds = int(raw_auth.get("refresh_skew_seconds") or 30)
        except (TypeError, ValueError):
            refresh_skew_seconds = 30
        supplier = build_oauth_headers_supplier(
            token_url=str(raw_auth.get("token_url") or "").strip() or None,
            client_id=str(raw_auth.get("client_id") or "").strip() or None,
            client_secret=str(raw_auth.get("client_secret") or "").strip() or None,
            scope=str(raw_auth.get("scope") or "").strip() or None,
            audience=str(raw_auth.get("audience") or "").strip() or None,
            grant_type=str(raw_auth.get("grant_type") or "").strip() or "client_credentials",
            refresh_skew_seconds=refresh_skew_seconds,
        )
        supplied = supplier()
        if inspect.isawaitable(supplied):
            supplied = await supplied
        return _coerce_headers(supplied)

    return {}


def _normalize_connection_config(server_config: Mapping[str, Any]) -> AdapterConnectionConfig:
    connection: AdapterConnectionConfig = {}
    for key in ("transport", "url", "command"):
        value = server_config.get(key)
        if isinstance(value, str) and value.strip():
            connection[key] = value.strip()

    args_value = server_config.get("args")
    if isinstance(args_value, Sequence) and not isinstance(args_value, (str, bytes)):
        connection["args"] = [str(item) for item in args_value]

    headers_value = server_config.get("headers")
    if isinstance(headers_value, Mapping):
        connection["headers"] = {
            str(key): str(value)
            for key, value in headers_value.items()
            if str(key).strip() and str(value).strip()
        }

    env_value = server_config.get("env")
    if isinstance(env_value, Mapping):
        connection["env"] = {
            str(key): str(value)
            for key, value in env_value.items()
            if str(key).strip() and str(value).strip()
        }

    # These are the optional fields accepted by FastMCP 4's canonical remote
    # and stdio server definitions. Transport-specific constructors expose a
    # wider Python-only API, but settings and per-run overrides are JSON data.
    passthrough_keys = (
        "auth",
        "timeout",
        "cwd",
        "keep_alive",
    )
    for key in passthrough_keys:
        if key in server_config and server_config[key] is not None:
            if key == "auth" and isinstance(server_config[key], Mapping):
                auth_type = str(server_config[key].get("type") or "").strip().lower()
                if auth_type in {"none", "bearer", "oauth_client_credentials"}:
                    continue
            cast(dict[str, Any], connection)[key] = server_config[key]

    return connection


def _move_success_error_to_warnings(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    returncode = normalized.get("returncode")
    error = normalized.get("error")
    is_success = returncode == 0 or returncode == "0"
    if not is_success or not isinstance(error, str) or not error.strip():
        return normalized

    warnings_raw = normalized.get("warnings")
    warnings: list[str] = []
    if isinstance(warnings_raw, Sequence) and not isinstance(warnings_raw, (str, bytes)):
        warnings = [str(item) for item in warnings_raw if str(item).strip()]
    if error not in warnings:
        warnings.append(error)
    normalized["warnings"] = warnings
    normalized.pop("error", None)
    return normalized


def _normalize_json_payload_text(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return text, False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return text, False
    if not isinstance(parsed, Mapping):
        return text, False
    normalized = _move_success_error_to_warnings(cast(Mapping[str, Any], parsed))
    if normalized == dict(parsed):
        return text, False
    return json.dumps(normalized, ensure_ascii=False), True


def _normalize_langchain_tool_result(result: Any) -> Any:
    """Normalize successful CLI-style payloads after LangChain conversion."""
    if not isinstance(result, tuple) or len(result) != 2:
        return result

    raw_content, raw_artifact = result
    content = list(raw_content) if isinstance(raw_content, list) else raw_content
    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            normalized_text, changed = _normalize_json_payload_text(text)
            if changed:
                content[index] = {**block, "text": normalized_text}

    artifact = dict(raw_artifact) if isinstance(raw_artifact, Mapping) else raw_artifact
    if isinstance(artifact, dict):
        structured = artifact.get("structured_content")
        if isinstance(structured, Mapping):
            artifact["structured_content"] = _move_success_error_to_warnings(structured)

    return content, artifact


def _normalize_adapter_tool(tool: BaseTool) -> BaseTool:
    """Attach the app's successful-result policy to a first-party MCP tool."""
    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        raise TypeError(
            f"MCPAdapter returned unsupported tool type {type(tool).__name__!r} for {tool.name!r}"
        )
    original = cast(Callable[..., Awaitable[Any]], tool.coroutine)

    async def call_tool(**arguments: Any) -> Any:
        return _normalize_langchain_tool_result(await original(**arguments))

    return tool.model_copy(update={"coroutine": call_tool})


async def build_adapter_server_configs(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, AdapterConnectionConfig]:
    settings = get_mcp_settings()
    if not settings.enable_mcp_tools:
        return {}

    override = _extract_config_override(run_config)
    configured = override if override is not None else get_mcp_servers_config()
    if not configured:
        return {}

    selected_keys = _select_server_keys(
        cast(Mapping[str, Mapping[str, Any]], configured),
        server_keys=server_keys,
        run_config=run_config,
    )
    if not selected_keys:
        return {}

    resolved: dict[str, AdapterConnectionConfig] = {}
    for key in selected_keys:
        if not isinstance(configured.get(key), Mapping):
            continue
        server_config = cast(Mapping[str, Any], configured[key])
        normalized = _normalize_connection_config(server_config)
        auth_headers = await _resolve_server_auth_headers(server_config)
        if auth_headers:
            existing_headers = normalized.get("headers", {})
            normalized["headers"] = {**existing_headers, **auth_headers}
        resolved[key] = normalized
    return resolved


def _create_client(
    connections: dict[str, AdapterConnectionConfig],
) -> MCPAdapter:
    # ClientGroup provides deterministic ``<server>_<tool>`` namespacing and
    # routes each wrapped tool to the client that advertised it.
    group = ClientGroup.from_config({"mcpServers": cast(dict[str, Any], connections)})
    return MCPAdapter(group)


async def _evict_failed_client(cache_key: str, client: MCPAdapter) -> None:
    _client_cache.pop(cache_key, None)
    _tool_cache.pop(cache_key, None)
    await _close_adapter(client)


async def _close_adapter(adapter: MCPAdapter) -> None:
    client_or_group = adapter.client
    clients = (
        list(client_or_group.clients.values())
        if isinstance(client_or_group, ClientGroup)
        else [client_or_group]
    )
    for client in clients:
        try:
            await client.close()
        except Exception as close_exc:  # noqa: BLE001
            logger.debug("MCP: FastMCP client cleanup failed: %s", close_exc)


def _connections_cache_key(connections: dict[str, AdapterConnectionConfig]) -> str:
    payload = {k: dict(v) for k, v in sorted(connections.items())}
    return json.dumps(payload, sort_keys=True, default=str)


async def load_adapter_tools(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    connections = await build_adapter_server_configs(server_keys=server_keys, run_config=run_config)
    if not connections:
        return []
    cache_key = _connections_cache_key(connections)
    async with _client_lock:
        cached_tools = _tool_cache.get(cache_key)
        if cached_tools is not None:
            logger.debug(
                "MCP: reusing cached tool list for %d server(s) (key hash=%s, tools=%d)",
                len(connections),
                hash(cache_key) & 0xFFFFFFFF,
                len(cached_tools),
            )
            return list(cached_tools)
        client = _client_cache.get(cache_key)
        if client is None:
            client = _create_client(connections)
            _client_cache[cache_key] = client
            logger.info(
                "MCP: cached MCPAdapter for %d server(s) (key hash=%s)",
                len(connections),
                hash(cache_key) & 0xFFFFFFFF,
            )
        try:
            tools = [_normalize_adapter_tool(tool) for tool in await client.list_tools()]
        except asyncio.CancelledError:
            # Evict before cleanup and re-raise cancellation so a partially started
            # client cannot be reused by a later tool load.
            await _evict_failed_client(cache_key, client)
            raise
        except Exception:
            # Avoid leaking broken clients/sessions in cache when adapter startup fails.
            await _evict_failed_client(cache_key, client)
            raise
        _tool_cache[cache_key] = list(tools)
        logger.info(
            "MCP: cached tool list for %d server(s) (key hash=%s, tools=%d)",
            len(connections),
            hash(cache_key) & 0xFFFFFFFF,
            len(tools),
        )
        return list(tools)


async def clear_adapter_runtime_cache() -> None:
    async with _client_lock:
        cached_adapters = list(_client_cache.values())
        _tool_cache.clear()
        _client_cache.clear()

    for adapter in cached_adapters:
        await _close_adapter(adapter)

    logger.info("MCP: cleared adapter runtime cache")


__all__ = [
    "AdapterConnectionConfig",
    "build_adapter_server_configs",
    "clear_adapter_runtime_cache",
    "load_adapter_tools",
]

"""App-side MCP client wiring around ``langchain_mcp_adapters``.

``MultiServerMCPClient`` owns MCP sessions, transports, and LangChain tool wrapping.
This module maps ``MCP_SERVERS_CONFIG`` + per-run ``RunnableConfig`` to connection
dicts, caches clients/tools, and must not reimplement MCP wire protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypedDict, cast

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from mcp.types import CallToolResult

from .mcp_settings import get_mcp_servers_config, get_mcp_settings

logger = logging.getLogger(__name__)

_client_lock = asyncio.Lock()
_client_cache: dict[str, MultiServerMCPClient] = {}
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
    sse_read_timeout: Any
    session_kwargs: dict[str, Any]
    httpx_client_factory: Any
    cwd: str
    encoding: str
    encoding_error_handler: str
    terminate_on_close: bool


def _extract_configurable(run_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(run_config, Mapping):
        return {}
    configurable = run_config.get("configurable")
    if isinstance(configurable, Mapping):
        return cast(Mapping[str, Any], configurable)
    return run_config


def _extract_server_keys_from_run_config(run_config: Mapping[str, Any] | None) -> list[str] | None:
    configurable = _extract_configurable(run_config)
    selected = configurable.get("mcp_server_keys")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        return None
    keys = [str(item).strip() for item in selected if str(item).strip()]
    return keys or None


def _extract_mcp_url_from_run_config(run_config: Mapping[str, Any] | None) -> str | None:
    configurable = _extract_configurable(run_config)
    mcp_url = configurable.get("mcp_url")
    if isinstance(mcp_url, str) and mcp_url.strip():
        return mcp_url.strip()
    return None


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

    mcp_url = _extract_mcp_url_from_run_config(run_config)
    if mcp_url:
        matched = [
            key
            for key, server_cfg in configured_servers.items()
            if isinstance(server_cfg.get("url"), str) and server_cfg.get("url") == mcp_url
        ]
        if matched:
            return matched
    return configured_keys


def _coerce_headers(raw_headers: object) -> dict[str, Any]:
    if not isinstance(raw_headers, Mapping):
        return {}
    return {
        str(key): value
        for key, value in raw_headers.items()
        if str(key).strip() and value is not None
    }


def _resolve_jwt_headers(settings: object) -> dict[str, str]:
    if not bool(getattr(settings, "enable_mcp_client_jwt", False)):
        return {}
    supplier = getattr(settings, "jwt_headers_supplier", None)
    if not callable(supplier):
        return {}
    try:
        supplied = supplier()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP: jwt header supplier failed: %s", exc)
        return {}
    return _coerce_headers(supplied)


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

    # Pass through adapter-supported optional fields to avoid dropping advanced
    # MCP connection options configured in settings.
    passthrough_keys = (
        "auth",
        "timeout",
        "sse_read_timeout",
        "session_kwargs",
        "httpx_client_factory",
        "cwd",
        "encoding",
        "encoding_error_handler",
        "terminate_on_close",
    )
    for key in passthrough_keys:
        if key in server_config and server_config[key] is not None:
            connection[key] = cast(Any, server_config[key])

    return connection


def _resolve_client_callbacks(settings: object) -> Any | None:
    supplier = getattr(settings, "mcp_client_callbacks_supplier", None)
    if callable(supplier):
        try:
            return supplier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP: callbacks supplier failed: %s", exc)
            return None
    return getattr(settings, "mcp_client_callbacks", None)


def _resolve_tool_interceptors(settings: object) -> list[Any] | None:
    interceptors: list[Any] = [_successful_tool_result_warning_interceptor]
    supplier = getattr(settings, "mcp_tool_interceptors_supplier", None)
    if callable(supplier):
        try:
            supplied = supplier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP: tool interceptors supplier failed: %s", exc)
            return interceptors
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            interceptors.extend([item for item in supplied])
        return interceptors

    raw = getattr(settings, "mcp_tool_interceptors", None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        interceptors.extend([item for item in raw])
    return interceptors


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


def _normalize_call_tool_result(result: MCPToolCallResult) -> MCPToolCallResult:
    if not isinstance(result, CallToolResult) or bool(result.isError):
        return result

    changed = False
    updates: dict[str, Any] = {}

    structured = result.structuredContent
    if isinstance(structured, Mapping):
        normalized_structured = _move_success_error_to_warnings(cast(Mapping[str, Any], structured))
        if normalized_structured != dict(structured):
            updates["structuredContent"] = normalized_structured
            changed = True

    normalized_content: list[Any] = []
    content_changed = False
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, Mapping):
            normalized_text = _move_success_error_to_warnings(cast(Mapping[str, Any], text))
            if normalized_text != dict(text):
                content_changed = True
                model_copy = getattr(block, "model_copy", None)
                if callable(model_copy):
                    normalized_content.append(model_copy(update={"text": normalized_text}))
                else:
                    normalized_content.append(block)
                continue
        if isinstance(text, str):
            normalized_text, did_change = _normalize_json_payload_text(text)
            if did_change:
                content_changed = True
                model_copy = getattr(block, "model_copy", None)
                if callable(model_copy):
                    normalized_content.append(model_copy(update={"text": normalized_text}))
                else:
                    normalized_content.append(block)
                continue
        normalized_content.append(block)

    if content_changed:
        updates["content"] = normalized_content
        changed = True

    if not changed:
        return result

    try:
        return result.model_copy(update=updates)
    except Exception:
        return result


async def _successful_tool_result_warning_interceptor(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
) -> MCPToolCallResult:
    result = await handler(request)
    return _normalize_call_tool_result(result)


def build_adapter_server_configs(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, AdapterConnectionConfig]:
    settings = get_mcp_settings()
    if not settings.enable_mcp_tools:
        return {}

    configured = get_mcp_servers_config()
    if not configured:
        return {}

    selected_keys = _select_server_keys(
        cast(Mapping[str, Mapping[str, Any]], configured),
        server_keys=server_keys,
        run_config=run_config,
    )
    if not selected_keys:
        return {}

    jwt_headers = _resolve_jwt_headers(settings)
    resolved: dict[str, AdapterConnectionConfig] = {}
    for key in selected_keys:
        if not isinstance(configured.get(key), Mapping):
            continue
        normalized = _normalize_connection_config(cast(Mapping[str, Any], configured[key]))
        if jwt_headers:
            existing_headers = normalized.get("headers", {})
            normalized["headers"] = {**existing_headers, **jwt_headers}
        resolved[key] = normalized
    return resolved


def _create_client(
    connections: dict[str, AdapterConnectionConfig],
    *,
    settings: object,
) -> MultiServerMCPClient:
    callbacks = _resolve_client_callbacks(settings)
    tool_interceptors = _resolve_tool_interceptors(settings)
    return MultiServerMCPClient(
        connections=cast(dict[str, Any], connections),
        callbacks=cast(Any, callbacks),
        tool_interceptors=cast(Any, tool_interceptors),
        tool_name_prefix=True,
    )


def _connections_cache_key(connections: dict[str, AdapterConnectionConfig]) -> str:
    payload = {k: dict(v) for k, v in sorted(connections.items())}
    return json.dumps(payload, sort_keys=True, default=str)


async def load_adapter_tools(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    settings = get_mcp_settings()
    connections = build_adapter_server_configs(server_keys=server_keys, run_config=run_config)
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
            client = _create_client(connections, settings=settings)
            _client_cache[cache_key] = client
            logger.info(
                "MCP: cached MultiServerMCPClient for %d server(s) (key hash=%s)",
                len(connections),
                hash(cache_key) & 0xFFFFFFFF,
            )
        try:
            tools = cast(list[BaseTool], await client.get_tools())
        except Exception:
            # Avoid leaking broken clients/sessions in cache when adapter startup fails.
            _client_cache.pop(cache_key, None)
            _tool_cache.pop(cache_key, None)
            for close_name in ("aclose", "close"):
                close_method = getattr(client, close_name, None)
                if not callable(close_method):
                    continue
                try:
                    close_result = close_method()
                    if inspect.isawaitable(close_result):
                        await cast(Awaitable[Any], close_result)
                except Exception as close_exc:  # noqa: BLE001
                    logger.debug(
                        "MCP: client cleanup via %s failed after get_tools error: %s",
                        close_name,
                        close_exc,
                    )
                break
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
        cached_tools = [tool for tools in _tool_cache.values() for tool in tools]
        _tool_cache.clear()
        _client_cache.clear()

    # Best-effort cleanup for any adapter tools holding closeable resources.
    for tool in cached_tools:
        for attr_name in ("aclose", "close"):
            close_method = getattr(tool, attr_name, None)
            if not callable(close_method):
                continue
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.debug("MCP: tool cleanup skipped for %s via %s: %s", tool.name, attr_name, exc)
            break

    logger.info("MCP: cleared adapter runtime cache")


__all__ = [
    "AdapterConnectionConfig",
    "build_adapter_server_configs",
    "clear_adapter_runtime_cache",
    "load_adapter_tools",
]

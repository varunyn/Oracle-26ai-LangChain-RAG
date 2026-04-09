from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from fastmcp import Client as FastMCPClient
from langchain_core.tools import BaseTool
from pydantic import Field
from typing_extensions import override

from .mcp_settings import get_mcp_servers_config, get_mcp_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolMetadata:
    canonical_name: str
    tool_name: str
    server_key: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "tool_name": self.tool_name,
            "server_key": self.server_key,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class _LoadedTool:
    tool: BaseTool
    metadata: MCPToolMetadata


def _run_coroutine_in_thread(coro: Coroutine[object, object, object]) -> object:
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("Thread runner did not return a value.")
    return result["value"]


def _canonical_tool_name(server_key: str, tool_name: str) -> str:
    return f"{server_key}.{tool_name}"


def _coerce_dict(raw: object) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(cast(Mapping[str, Any], raw))
    return {}


def _coerce_tool_schema(tool_obj: object) -> dict[str, Any]:
    for attr in ("inputSchema", "input_schema"):
        candidate = getattr(tool_obj, attr, None)
        if isinstance(candidate, Mapping):
            return dict(cast(Mapping[str, Any], candidate))
    return {}


def _coerce_tool_description(tool_obj: object) -> str:
    description = getattr(tool_obj, "description", None)
    if isinstance(description, str):
        return description.strip()
    return ""


def _coerce_tool_name(tool_obj: object) -> str | None:
    tool_name = getattr(tool_obj, "name", None)
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return None


def _serialize_tool_result(result: object) -> object:
    if result is None:
        return None
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            return cast(object, model_dump(exclude_none=True))
        except Exception:  # noqa: BLE001
            return str(result)
    return str(result)


class DirectMCPTool(BaseTool):
    name: str = Field(default="")
    description: str = ""
    server_key: str
    mcp_tool_name: str
    server_config: dict[str, Any]
    mcp_input_schema: dict[str, Any] = {}

    @override
    def _run(self, **kwargs: Any) -> object:
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._arun(**kwargs))
        return _run_coroutine_in_thread(self._arun(**kwargs))

    @override
    async def _arun(self, **kwargs: Any) -> object:
        client = _build_fastmcp_client(self.server_key, self.server_config)
        async with client:
            result = await client.call_tool(self.mcp_tool_name, arguments=kwargs or None)
        return _serialize_tool_result(result)


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

    settings_keys = get_mcp_settings().mcp_server_keys
    if settings_keys:
        filtered = [key for key in settings_keys if key in configured_servers]
        if filtered:
            return filtered

    if "default" in configured_servers:
        return ["default"]
    return configured_keys


def _build_fastmcp_client(
    server_key: str,
    server_config: Mapping[str, Any],
):
    url = server_config.get("url")
    if isinstance(url, str) and url.strip():
        return FastMCPClient(url.strip(), name=f"direct-mcp-{server_key}")
    return FastMCPClient(dict(server_config), name=f"direct-mcp-{server_key}")


async def _load_tools_for_server_async(
    server_key: str,
    server_config: Mapping[str, Any],
) -> list[_LoadedTool]:
    client = _build_fastmcp_client(server_key, server_config)
    async with client:
        tool_objects = await client.list_tools()

    loaded: list[_LoadedTool] = []
    for tool_obj in cast(Sequence[object], tool_objects or []):
        tool_name = _coerce_tool_name(tool_obj)
        if tool_name is None:
            continue
        canonical_name = _canonical_tool_name(server_key, tool_name)
        description = _coerce_tool_description(tool_obj)
        input_schema = _coerce_tool_schema(tool_obj)
        wrapped_tool = DirectMCPTool(
            name=canonical_name,
            description=description,
            server_key=server_key,
            mcp_tool_name=tool_name,
            server_config=_coerce_dict(server_config),
            mcp_input_schema=input_schema,
        )
        loaded.append(
            _LoadedTool(
                tool=wrapped_tool,
                metadata=MCPToolMetadata(
                    canonical_name=canonical_name,
                    tool_name=tool_name,
                    server_key=server_key,
                    description=description,
                    input_schema=input_schema,
                ),
            )
        )
    return loaded


async def _load_catalog_async(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[_LoadedTool]:
    settings = get_mcp_settings()
    if not settings.enable_mcp_tools:
        return []

    configured = get_mcp_servers_config()
    if not configured:
        return []

    selected_keys = _select_server_keys(
        cast(Mapping[str, Mapping[str, Any]], configured),
        server_keys=server_keys,
        run_config=run_config,
    )
    if not selected_keys:
        return []

    catalog: list[_LoadedTool] = []
    for server_key in selected_keys:
        server_config = configured.get(server_key)
        if not isinstance(server_config, Mapping):
            logger.warning("Direct MCP tools: server config for key '%s' is invalid", server_key)
            continue
        try:
            loaded_for_server = await _load_tools_for_server_async(server_key, server_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Direct MCP tools: failed loading tools for server '%s': %s",
                server_key,
                exc,
            )
            continue
        catalog.extend(loaded_for_server)
    return catalog


async def get_mcp_tools_async(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    catalog = await _load_catalog_async(server_keys=server_keys, run_config=run_config)
    return [entry.tool for entry in catalog]


def get_mcp_tools(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    coro = get_mcp_tools_async(server_keys=server_keys, run_config=run_config)
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return cast(list[BaseTool], asyncio.run(coro))
    return cast(list[BaseTool], _run_coroutine_in_thread(coro))


async def get_mcp_tool_metadata_async(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    catalog = await _load_catalog_async(server_keys=server_keys, run_config=run_config)
    return [entry.metadata.to_dict() for entry in catalog]


def get_mcp_tool_metadata(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    coro = get_mcp_tool_metadata_async(server_keys=server_keys, run_config=run_config)
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return cast(list[dict[str, Any]], asyncio.run(coro))
    return cast(list[dict[str, Any]], _run_coroutine_in_thread(coro))


__all__ = [
    "MCPToolMetadata",
    "DirectMCPTool",
    "get_mcp_tools",
    "get_mcp_tools_async",
    "get_mcp_tool_metadata",
    "get_mcp_tool_metadata_async",
]

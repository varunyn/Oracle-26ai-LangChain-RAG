from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.tools import BaseTool

from .mcp_adapter_runtime import load_adapter_tools
from .mcp_settings import get_mcp_servers_config

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


def _schema_from_tool(tool: BaseTool) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, Mapping):
        return dict(cast(Mapping[str, Any], args_schema))
    if args_schema is None:
        return {}
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
        if isinstance(schema, Mapping):
            return dict(cast(Mapping[str, Any], schema))
    schema_attr = getattr(tool, "args", None)
    if isinstance(schema_attr, Mapping):
        return dict(cast(Mapping[str, Any], schema_attr))
    return {}


def _metadata_from_tools(tools: Sequence[BaseTool]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for tool in tools:
        canonical_name = str(getattr(tool, "name", "") or "").strip()
        if not canonical_name:
            continue
        if "." in canonical_name:
            server_key, tool_name = canonical_name.split(".", 1)
        else:
            server_key, tool_name = "", canonical_name
        metadata.append(
            MCPToolMetadata(
                canonical_name=canonical_name,
                tool_name=tool_name,
                server_key=server_key,
                description=str(getattr(tool, "description", "") or "").strip(),
                input_schema=_schema_from_tool(tool),
            ).to_dict()
        )
    return metadata


async def get_mcp_tools_async(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    requested_keys = [str(key).strip() for key in (server_keys or []) if str(key).strip()]
    configured = get_mcp_servers_config() or {}
    configured_keys = [str(key).strip() for key in configured.keys() if str(key).strip()]
    target_keys = requested_keys or configured_keys

    try:
        return await load_adapter_tools(server_keys=server_keys, run_config=run_config)
    except Exception:  # noqa: BLE001
        logger.exception("Direct MCP tools: failed loading adapter tools (all servers)")
        if len(target_keys) <= 1:
            return []

        # Degrade gracefully: one broken MCP server should not hide tools from healthy servers.
        tools: list[BaseTool] = []
        seen_names: set[str] = set()
        for key in target_keys:
            try:
                partial = await load_adapter_tools(server_keys=[key], run_config=run_config)
            except Exception:  # noqa: BLE001
                logger.exception("Direct MCP tools: failed loading server '%s'", key)
                continue
            for tool in partial:
                tool_name = str(getattr(tool, "name", "") or "").strip()
                if not tool_name or tool_name in seen_names:
                    continue
                seen_names.add(tool_name)
                tools.append(tool)

        if tools:
            logger.warning(
                "Direct MCP tools: recovered partial tool set after failures (servers=%d, tools=%d)",
                len(target_keys),
                len(tools),
            )
        else:
            logger.warning(
                "Direct MCP tools: no MCP tools available after per-server recovery attempts (servers=%d)",
                len(target_keys),
            )
        return tools


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
    tools = await get_mcp_tools_async(server_keys=server_keys, run_config=run_config)
    return _metadata_from_tools(tools)


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
    "get_mcp_tools",
    "get_mcp_tools_async",
    "get_mcp_tool_metadata",
    "get_mcp_tool_metadata_async",
    "load_adapter_tools",
]

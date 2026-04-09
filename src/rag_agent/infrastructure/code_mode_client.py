# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Mapping
from typing import cast

from utcp.data.call_template import CallTemplate, CallTemplateSerializer  # type: ignore
from utcp_code_mode import CodeModeUtcpClient  # type: ignore

from .mcp_settings import get_mcp_servers_config, normalize_mcp_transport

logger = logging.getLogger(__name__)

_code_mode_client: CodeModeUtcpClient | None = None

MCPServersConfig = dict[str, object]


async def init_code_mode_client() -> None:
    global _code_mode_client
    if _code_mode_client is not None:
        return

    servers = cast(MCPServersConfig, get_mcp_servers_config())
    if not servers:
        logger.warning("Code-mode MCP: no MCP servers configured; MCP disabled")
        return

    client: CodeModeUtcpClient | None = None
    try:
        client = await CodeModeUtcpClient.create()
        await _register_mcp_bundle(client, servers)
        try:
            tools = await client.get_tools()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Code-mode MCP: readiness check failed; MCP disabled: %s",
                exc,
                exc_info=True,
            )
            await _aclose_client(client)
            return
        if not tools:
            logger.warning("Code-mode MCP: readiness check found no tools; MCP disabled")
            await _aclose_client(client)
            return
        _code_mode_client = client
        logger.info("Code-mode MCP: registered %d MCP servers", len(servers))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Code-mode MCP: failed to initialize; MCP disabled: %s",
            exc,
            exc_info=True,
        )
        if client is not None:
            await _aclose_client(client)


def get_code_mode_client() -> CodeModeUtcpClient:
    if _code_mode_client is None:
        raise RuntimeError(
            "Code-mode MCP client not initialized. Ensure init_code_mode_client() runs at startup and MCP_SERVERS_CONFIG is configured."
        )
    return _code_mode_client


def _python_safe_tool_name(full_name: str) -> str:
    """Convert tool name like mcp_bundle.oci-mcp-server.get_oci_command_help to the identifier
    used in the code-mode execution context (hyphens and other non-identifier chars → underscore).
    Must match utcp_code_mode's _sanitize_identifier + manual.attr logic so generated code runs.
    """
    parts = full_name.split(".")
    if not parts:
        return full_name

    # Same rule as typical sanitize: only [a-zA-Z0-9_]
    def sanitize(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", value)

    manual = sanitize(parts[0])
    if len(parts) == 1:
        return manual
    # tool_parts joined with underscore (how the library sets the attribute on Manual)
    attr = "_".join(sanitize(p) for p in parts[1:])
    return f"{manual}.{attr}"


async def get_tool_summary() -> str:
    """Return a short summary of registered MCP tools (name + description) for prompt injection.
    Emits Python-valid names so the model writes runnable code (no hyphens; sandbox has no []).
    """
    client = _code_mode_client
    if client is None:
        return ""
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Code-mode MCP: could not get tools for summary: %s", exc)
        return ""
    if not tools:
        return ""
    lines = [
        "Available tools — use the exact names below in your code (they are valid Python identifiers):",
        "",
    ]
    for t in tools:
        desc = (getattr(t, "description", None) or "").strip() or "(no description)"
        python_name = _python_safe_tool_name(t.name)
        lines.append(f"- {python_name}: {desc}")
    return "\n".join(lines)


async def get_tool_name_map() -> dict[str, str]:
    client = _code_mode_client
    if client is None:
        return {}
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Code-mode MCP: could not get tools for name map: %s", exc)
        return {}
    if not tools:
        return {}
    mapping: dict[str, str] = {}
    for t in tools:
        mapping[_python_safe_tool_name(t.name)] = t.name
    return mapping


async def aclose_code_mode_client() -> None:
    global _code_mode_client
    client = _code_mode_client
    if client is None:
        return
    try:
        await _aclose_client(client)
    except Exception:  # noqa: BLE001
        logger.debug("Code-mode MCP: error during client close", exc_info=True)
    finally:
        _code_mode_client = None


async def _aclose_client(client: CodeModeUtcpClient) -> None:
    close_method = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close_method is None:
        return
    result_obj = cast(object, close_method())
    if inspect.isawaitable(result_obj):
        _ = await cast(Awaitable[object], result_obj)


async def _register_mcp_bundle(
    client: CodeModeUtcpClient,
    servers: MCPServersConfig,
) -> None:
    normalized = _normalize_mcp_servers_config(servers)
    try:
        call_template = _build_call_template(normalized)
        _ = await client.register_manual(call_template)
    except Exception as exc:  # noqa: BLE001
        if not _should_retry_transport_fallback(exc):
            raise
        fallback_servers, changed = _map_transport_fallback(normalized)
        if not changed:
            raise
        logger.info("Code-mode MCP: retrying registration with transport fallback to http")
        call_template = _build_call_template(fallback_servers)
        _ = await client.register_manual(call_template)


def _build_call_template(servers: MCPServersConfig) -> CallTemplate:
    return CallTemplateSerializer().validate_dict(  # type: ignore
        {
            "name": "mcp_bundle",
            "call_template_type": "mcp",
            "config": {"mcpServers": servers},
        }
    )


def _should_retry_transport_fallback(exc: Exception) -> bool:
    message = str(exc).lower()
    return "transport" in message or "streamable" in message


def _map_transport_fallback(
    servers: MCPServersConfig,
) -> tuple[MCPServersConfig, bool]:
    changed = False
    mapped: MCPServersConfig = {}
    for name, server in servers.items():
        if not isinstance(server, Mapping):
            mapped[name] = server
            continue
        server_map = cast(Mapping[str, object], server)
        transport = normalize_mcp_transport(server_map.get("transport"))
        if transport == "streamable-http":
            updated = dict(server_map)
            updated["transport"] = "http"
            mapped[name] = updated
            changed = True
        else:
            mapped[name] = server_map
    return mapped, changed


def _normalize_mcp_servers_config(servers: MCPServersConfig) -> MCPServersConfig:
    normalized: MCPServersConfig = {}
    for name, server in servers.items():
        if not isinstance(server, Mapping):
            normalized[name] = server
            continue
        server_map = dict(cast(Mapping[str, object], server))
        server_map["transport"] = normalize_mcp_transport(server_map.get("transport"))
        if "auth" not in server_map:
            server_map["auth"] = None
        normalized[name] = server_map
    return normalized

# MCP Client Settings Module
"""
Configuration for when this app **consumes** MCP (acts as an MCP client).

Use this only for: RAG Answer node calling external MCP servers to get tools.
Settings (env or .env): ENABLE_MCP_TOOLS and optional MCP_SERVERS_CONFIG seed
config. UI-managed MCP server config is stored at MCP_UI_CONFIG_FILE. See
docs/CONFIGURATION.md.
"""

import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from .mcp_config_store import resolve_enabled_mcp_servers_config


def get_mcp_servers_config() -> dict[str, dict[str, Any]]:
    """Return enabled MCP server config from UI store, or optional env seed."""
    from api.settings import get_settings

    settings = get_settings()
    resolved = resolve_enabled_mcp_servers_config(
        base_config=settings.MCP_SERVERS_CONFIG,
        store_path=settings.MCP_UI_CONFIG_FILE,
    )
    return _normalize_mcp_server_urls(resolved)


def _is_running_in_docker() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as handle:
            contents = handle.read()
        return "docker" in contents or "containerd" in contents
    except Exception:
        return False


def _normalize_mcp_server_urls(
    cfg: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if _is_running_in_docker():
        return cfg
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in cfg.items():
        if not isinstance(entry, dict):
            normalized[name] = entry
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            normalized[name] = dict(entry)
            continue
        parsed = urlparse(url)
        if parsed.hostname != "host.docker.internal":
            normalized[name] = dict(entry)
            continue
        netloc = f"localhost:{parsed.port or 80}"
        updated = dict(entry)
        updated["url"] = urlunparse(parsed._replace(netloc=netloc))
        normalized[name] = updated
    return normalized


class MCPSettings:
    """MCP **client** settings: when this app consumes MCP tools (no auth by default)."""

    def __init__(self) -> None:
        from api.settings import get_settings

        self.enable_mcp_tools = get_settings().ENABLE_MCP_TOOLS


def get_mcp_settings() -> MCPSettings:
    """Return MCP client settings instance."""
    return MCPSettings()


__all__ = [
    "MCPSettings",
    "get_mcp_servers_config",
    "get_mcp_settings",
]

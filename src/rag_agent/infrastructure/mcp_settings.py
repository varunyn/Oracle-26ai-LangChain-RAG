# MCP Client Settings Module
"""
Configuration for when this app **consumes** MCP (acts as an MCP client).

Use this only for: RAG Answer node calling external MCP servers to get tools.
When this app **runs as** an MCP server (mcp_servers/*.py), TRANSPORT and PORT
come from .env / get_settings().

Settings (env or .env): ENABLE_MCP_TOOLS, MCP_SERVER_KEYS (e.g. default,context7),
ENABLE_MCP_CLIENT_JWT. See docs/CONFIGURATION.md and .env.example.
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from .mcp_oauth import build_oauth_headers_supplier

logger = logging.getLogger(__name__)


def normalize_mcp_transport(transport: object) -> str:
    """Normalize legacy MCP transport spellings to the canonical FastMCP v3 form."""
    value = str(transport or "streamable-http").strip().lower()
    if value in {"streamable_http", "streamable-http", "http"}:
        return "streamable-http"
    if value == "stdio":
        return "stdio"
    return value or "streamable-http"


def _default_enable_mcp_tools() -> bool:
    """Default from settings when env ENABLE_MCP_TOOLS is not set."""
    try:
        from api.settings import get_settings

        return bool(getattr(get_settings(), "ENABLE_MCP_TOOLS", True))
    except Exception as e:
        logger.debug("MCP: settings unavailable, defaulting enable_mcp_tools=True: %s", e)
        return True


def _get_app_settings() -> Any | None:
    try:
        from api.settings import get_settings

        return get_settings()
    except Exception as e:
        logger.debug("MCP: app settings unavailable: %s", e)
        return None


def get_mcp_servers_config() -> dict[str, dict[str, Any]]:
    """Return MCP server config from settings (single source of truth)."""
    try:
        from api.settings import get_settings

        cfg = getattr(get_settings(), "MCP_SERVERS_CONFIG", None)
        if not isinstance(cfg, dict):
            logger.debug(
                "MCP: MCP_SERVERS_CONFIG missing or invalid; expected dict, got %s", type(cfg)
            )
            return {}
        return _normalize_mcp_server_urls(_normalize_mcp_server_transports(cfg))
    except Exception as e:
        logger.debug("MCP: settings unavailable; no servers configured: %s", e)
        return {}


def _is_running_in_docker() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as handle:
            contents = handle.read()
        return "docker" in contents or "containerd" in contents
    except Exception:
        return False


def _normalize_mcp_server_transports(
    cfg: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in cfg.items():
        if not isinstance(entry, dict):
            normalized[name] = entry
            continue
        updated = dict(entry)
        updated["transport"] = normalize_mcp_transport(updated.get("transport"))
        normalized[name] = updated
    return normalized


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
        app_settings = _get_app_settings()
        env_val = os.getenv("ENABLE_MCP_TOOLS", "").strip().lower()
        if env_val in ("true", "false"):
            self.enable_mcp_tools = env_val == "true"
        else:
            self.enable_mcp_tools = _default_enable_mcp_tools()
        raw = os.getenv("MCP_SERVER_KEYS", "").strip()
        if raw:
            self.mcp_server_keys: list[str] | None = [
                k.strip() for k in raw.split(",") if k.strip()
            ]
        else:
            self.mcp_server_keys = None  # None = use "default" only
        # Client-only: send JWT when calling MCP servers that require auth (default False = no auth)
        self.enable_mcp_client_jwt: bool = (
            os.getenv("ENABLE_MCP_CLIENT_JWT", "false").lower() == "true"
        )
        self.jwt_headers_supplier = None  # Optional; used only if enable_mcp_client_jwt is True
        enable_mcp_oauth = bool(getattr(app_settings, "ENABLE_MCP_OAUTH", False))
        if self.enable_mcp_client_jwt and enable_mcp_oauth:
            self.jwt_headers_supplier = build_oauth_headers_supplier(
                token_url=getattr(app_settings, "MCP_OAUTH_TOKEN_URL", None),
                client_id=getattr(app_settings, "MCP_OAUTH_CLIENT_ID", None),
                client_secret=getattr(app_settings, "MCP_OAUTH_CLIENT_SECRET", None),
                scope=getattr(app_settings, "MCP_OAUTH_SCOPE", None),
                audience=getattr(app_settings, "MCP_OAUTH_AUDIENCE", None),
                grant_type=getattr(app_settings, "MCP_OAUTH_GRANT_TYPE", "client_credentials"),
                refresh_skew_seconds=int(
                    getattr(app_settings, "MCP_OAUTH_REFRESH_SKEW_SECONDS", 30)
                ),
            )
        self.mcp_client_callbacks = None
        self.mcp_tool_interceptors = None
        self.mcp_client_callbacks_supplier = None
        self.mcp_tool_interceptors_supplier = None


def get_mcp_settings() -> MCPSettings:
    """Return MCP client settings instance."""
    return MCPSettings()


# Default instance for import
mcp_settings = get_mcp_settings()


__all__ = [
    "MCPSettings",
    "get_mcp_servers_config",
    "get_mcp_settings",
    "mcp_settings",
    "normalize_mcp_transport",
]

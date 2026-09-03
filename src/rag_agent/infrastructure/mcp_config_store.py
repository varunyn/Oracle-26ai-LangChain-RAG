"""Server-side MCP configuration store used by the settings UI."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MCP_UI_CONFIG_FILE = ".local-data/mcp_servers.json"
SUPPORTED_TRANSPORTS = {"http", "streamable-http", "stdio"}
SUPPORTED_AUTH_TYPES = {"none", "bearer", "oauth_client_credentials"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MCPServerAuthConfig:
    type: str = "none"
    bearer_token: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    scope: str | None = None
    audience: str | None = None
    grant_type: str = "client_credentials"
    refresh_skew_seconds: int = 30


@dataclass(frozen=True)
class MCPServerConfig:
    key: str
    transport: str
    url: str
    enabled: bool = True
    auth: MCPServerAuthConfig = field(default_factory=MCPServerAuthConfig)


def resolve_store_path(path: str | os.PathLike[str] | None) -> Path:
    candidate = Path(path or DEFAULT_MCP_UI_CONFIG_FILE).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _normalize_auth_type(value: object) -> str:
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in SUPPORTED_AUTH_TYPES else "none"


def _normalize_optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_auth_config(raw_auth: object) -> MCPServerAuthConfig:
    if not isinstance(raw_auth, Mapping):
        return MCPServerAuthConfig()

    auth_type = _normalize_auth_type(raw_auth.get("type"))
    try:
        refresh_skew_seconds = int(raw_auth.get("refresh_skew_seconds") or 30)
    except (TypeError, ValueError):
        refresh_skew_seconds = 30
    if refresh_skew_seconds <= 0:
        refresh_skew_seconds = 30

    if auth_type == "bearer":
        return MCPServerAuthConfig(
            type="bearer",
            bearer_token=_normalize_optional_str(raw_auth.get("bearer_token")),
        )

    if auth_type == "oauth_client_credentials":
        return MCPServerAuthConfig(
            type="oauth_client_credentials",
            token_url=_normalize_optional_str(raw_auth.get("token_url")),
            client_id=_normalize_optional_str(raw_auth.get("client_id")),
            client_secret=_normalize_optional_str(raw_auth.get("client_secret")),
            scope=_normalize_optional_str(raw_auth.get("scope")),
            audience=_normalize_optional_str(raw_auth.get("audience")),
            grant_type=_normalize_optional_str(raw_auth.get("grant_type")) or "client_credentials",
            refresh_skew_seconds=refresh_skew_seconds,
        )

    return MCPServerAuthConfig()


def _auth_to_runtime_config(auth: MCPServerAuthConfig) -> dict[str, object]:
    if auth.type == "bearer":
        return {
            "type": "bearer",
            "bearer_token": auth.bearer_token,
        }
    if auth.type == "oauth_client_credentials":
        return {
            "type": "oauth_client_credentials",
            "token_url": auth.token_url,
            "client_id": auth.client_id,
            "client_secret": auth.client_secret,
            "scope": auth.scope,
            "audience": auth.audience,
            "grant_type": auth.grant_type,
            "refresh_skew_seconds": auth.refresh_skew_seconds,
        }
    return {"type": "none"}


def _normalize_base_config(
    base_config: Mapping[str, Mapping[str, Any]] | None,
) -> list[MCPServerConfig]:
    servers: list[MCPServerConfig] = []
    for key, value in (base_config or {}).items():
        normalized_key = str(key).strip()
        if not normalized_key or not isinstance(value, Mapping):
            continue
        transport = str(value.get("transport") or "streamable-http").strip()
        url = str(value.get("url") or "").strip()
        enabled_raw = value.get("enabled", True)
        if not url:
            continue
        servers.append(
            MCPServerConfig(
                key=normalized_key,
                transport=transport or "streamable-http",
                url=url,
                enabled=bool(enabled_raw),
                auth=_normalize_auth_config(value.get("auth")),
            )
        )
    return servers


def _read_store(path: Path) -> list[MCPServerConfig] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP UI config store could not be read from %s: %s", path, exc)
        return None

    raw_servers = data.get("servers") if isinstance(data, Mapping) else None
    if not isinstance(raw_servers, list):
        logger.warning("MCP UI config store has invalid shape at %s", path)
        return None

    servers: list[MCPServerConfig] = []
    for item in raw_servers:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "").strip()
        transport = str(item.get("transport") or "streamable-http").strip()
        url = str(item.get("url") or "").strip()
        if not key or not url:
            continue
        servers.append(
            MCPServerConfig(
                key=key,
                transport=transport or "streamable-http",
                url=url,
                enabled=bool(item.get("enabled", True)),
                auth=_normalize_auth_config(item.get("auth")),
            )
        )
    return servers


def _write_store(path: Path, servers: list[MCPServerConfig]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"servers": [asdict(server) for server in servers]}
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _materialize_servers(
    *,
    base_config: Mapping[str, Mapping[str, Any]] | None,
    store_path: str | os.PathLike[str] | Path | None,
) -> list[MCPServerConfig]:
    path = resolve_store_path(store_path)
    stored = _read_store(path)
    if stored is not None:
        return stored
    return _normalize_base_config(base_config)


def list_mcp_server_configs(
    *,
    base_config: Mapping[str, Mapping[str, Any]] | None,
    store_path: str | os.PathLike[str] | Path | None,
) -> list[MCPServerConfig]:
    return _materialize_servers(base_config=base_config, store_path=store_path)


def upsert_mcp_server_config(
    server: MCPServerConfig,
    *,
    base_config: Mapping[str, Mapping[str, Any]] | None,
    store_path: str | os.PathLike[str] | Path | None,
) -> MCPServerConfig:
    servers = _materialize_servers(base_config=base_config, store_path=store_path)
    next_servers = [item for item in servers if item.key != server.key]
    inserted = False
    for index, item in enumerate(servers):
        if item.key == server.key:
            next_servers.insert(index, server)
            inserted = True
            break
    if not inserted:
        next_servers.append(server)
    _write_store(resolve_store_path(store_path), next_servers)
    return server


def delete_mcp_server_config(
    key: str,
    *,
    base_config: Mapping[str, Mapping[str, Any]] | None,
    store_path: str | os.PathLike[str] | Path | None,
) -> None:
    normalized_key = key.strip()
    servers = _materialize_servers(base_config=base_config, store_path=store_path)
    next_servers = [item for item in servers if item.key != normalized_key]
    _write_store(resolve_store_path(store_path), next_servers)


def resolve_enabled_mcp_servers_config(
    *,
    base_config: Mapping[str, Mapping[str, Any]] | None,
    store_path: str | os.PathLike[str] | Path | None,
) -> dict[str, dict[str, Any]]:
    servers = _materialize_servers(base_config=base_config, store_path=store_path)
    resolved: dict[str, dict[str, Any]] = {}
    for server in servers:
        if not server.enabled:
            continue
        config: dict[str, Any] = {"transport": server.transport, "url": server.url}
        if server.auth.type != "none":
            config["auth"] = _auth_to_runtime_config(server.auth)
        resolved[server.key] = config
    return resolved


__all__ = [
    "DEFAULT_MCP_UI_CONFIG_FILE",
    "MCPServerAuthConfig",
    "MCPServerConfig",
    "SUPPORTED_AUTH_TYPES",
    "SUPPORTED_TRANSPORTS",
    "delete_mcp_server_config",
    "list_mcp_server_configs",
    "resolve_enabled_mcp_servers_config",
    "resolve_store_path",
    "upsert_mcp_server_config",
]

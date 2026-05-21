"""Server-side MCP configuration store used by the settings UI."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MCP_UI_CONFIG_FILE = ".local-data/mcp_servers.json"
SUPPORTED_TRANSPORTS = {"streamable-http", "sse", "stdio"}


@dataclass(frozen=True)
class MCPServerConfig:
    key: str
    transport: str
    url: str
    enabled: bool = True


def resolve_store_path(path: str | os.PathLike[str] | None) -> Path:
    candidate = Path(path or DEFAULT_MCP_UI_CONFIG_FILE).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


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
) -> dict[str, dict[str, str]]:
    servers = _materialize_servers(base_config=base_config, store_path=store_path)
    return {
        server.key: {"transport": server.transport, "url": server.url}
        for server in servers
        if server.enabled
    }


__all__ = [
    "DEFAULT_MCP_UI_CONFIG_FILE",
    "MCPServerConfig",
    "SUPPORTED_TRANSPORTS",
    "delete_mcp_server_config",
    "list_mcp_server_configs",
    "resolve_enabled_mcp_servers_config",
    "resolve_store_path",
    "upsert_mcp_server_config",
]

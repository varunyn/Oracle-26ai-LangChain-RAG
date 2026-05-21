"""Runtime config endpoints exposed on the shared API surface."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from api.deps.request import get_settings as get_settings_dep
from api.settings import Settings
from src.rag_agent.infrastructure.mcp_config_store import (
    SUPPORTED_TRANSPORTS,
    MCPServerConfig,
    delete_mcp_server_config,
    list_mcp_server_configs,
    upsert_mcp_server_config,
)

router = APIRouter(prefix="/api", tags=["config"])
_MCP_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class MCPServerConfigResponse(BaseModel):
    key: str
    transport: str
    url: str
    enabled: bool


class MCPServersConfigResponse(BaseModel):
    enable_mcp_tools: bool
    servers: list[MCPServerConfigResponse]


class MCPServerConfigWrite(BaseModel):
    transport: str = Field(default="streamable-http")
    url: str
    enabled: bool = True

    @field_validator("transport")
    @classmethod
    def _validate_transport(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                "transport must be one of: " + ", ".join(sorted(SUPPORTED_TRANSPORTS))
            )
        return normalized

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("url is required")
        return normalized


class MCPServerEnabledWrite(BaseModel):
    enabled: bool


@router.get("/config")
async def get_config(request: Request) -> dict[str, object]:
    settings: Settings = get_settings_dep(request)
    return {
        "region": settings.REGION,
        "embed_model_id": settings.EMBED_MODEL_ID,
        "model_list": settings.MODEL_LIST,
        "model_display_names": settings.MODEL_DISPLAY_NAMES,
        "collection_list": settings.COLLECTION_LIST or [settings.DEFAULT_COLLECTION],
        "enable_user_feedback": settings.ENABLE_USER_FEEDBACK,
    }


def _settings_mcp_config(settings: Settings) -> dict[str, dict[str, str]]:
    raw_config = getattr(settings, "MCP_SERVERS_CONFIG", None)
    if not isinstance(raw_config, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in raw_config.items()
        if isinstance(value, dict)
    }


def _store_path(settings: Settings) -> str:
    return str(getattr(settings, "MCP_UI_CONFIG_FILE", ".local-data/mcp_servers.json"))


def _validate_mcp_key(key: str) -> str:
    normalized = key.strip()
    if not _MCP_KEY_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=422,
            detail="MCP server key must be 1-64 characters using letters, numbers, _, -, or .",
        )
    return normalized


def _to_response(server: MCPServerConfig) -> MCPServerConfigResponse:
    return MCPServerConfigResponse(
        key=server.key,
        transport=server.transport,
        url=server.url,
        enabled=server.enabled,
    )


@router.get("/config/mcp-servers", response_model=MCPServersConfigResponse)
async def get_mcp_servers_config_api(request: Request) -> MCPServersConfigResponse:
    settings: Settings = get_settings_dep(request)
    servers = list_mcp_server_configs(
        base_config=_settings_mcp_config(settings),
        store_path=_store_path(settings),
    )
    return MCPServersConfigResponse(
        enable_mcp_tools=bool(getattr(settings, "ENABLE_MCP_TOOLS", True)),
        servers=[_to_response(server) for server in servers],
    )


@router.put("/config/mcp-servers/{key}", response_model=MCPServerConfigResponse)
async def put_mcp_server_config_api(
    key: str,
    payload: MCPServerConfigWrite,
    request: Request,
) -> MCPServerConfigResponse:
    settings: Settings = get_settings_dep(request)
    server = upsert_mcp_server_config(
        MCPServerConfig(
            key=_validate_mcp_key(key),
            transport=payload.transport,
            url=payload.url,
            enabled=payload.enabled,
        ),
        base_config=_settings_mcp_config(settings),
        store_path=_store_path(settings),
    )
    return _to_response(server)


@router.patch("/config/mcp-servers/{key}/enabled", response_model=MCPServerConfigResponse)
async def patch_mcp_server_enabled_api(
    key: str,
    payload: MCPServerEnabledWrite,
    request: Request,
) -> MCPServerConfigResponse:
    settings: Settings = get_settings_dep(request)
    normalized_key = _validate_mcp_key(key)
    servers = list_mcp_server_configs(
        base_config=_settings_mcp_config(settings),
        store_path=_store_path(settings),
    )
    current = next((server for server in servers if server.key == normalized_key), None)
    if current is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    updated = MCPServerConfig(
        key=current.key,
        transport=current.transport,
        url=current.url,
        enabled=payload.enabled,
    )
    return _to_response(
        upsert_mcp_server_config(
            updated,
            base_config=_settings_mcp_config(settings),
            store_path=_store_path(settings),
        )
    )


@router.delete("/config/mcp-servers/{key}")
async def delete_mcp_server_config_api(key: str, request: Request) -> dict[str, object]:
    settings: Settings = get_settings_dep(request)
    delete_mcp_server_config(
        _validate_mcp_key(key),
        base_config=_settings_mcp_config(settings),
        store_path=_store_path(settings),
    )
    return {"ok": True}

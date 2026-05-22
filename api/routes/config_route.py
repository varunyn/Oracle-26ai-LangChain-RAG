"""Runtime config endpoints exposed on the shared API surface."""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from api.deps.request import get_settings as get_settings_dep
from api.settings import Settings
from src.rag_agent.infrastructure import mcp_adapter_runtime
from src.rag_agent.infrastructure.mcp_config_store import (
    SUPPORTED_AUTH_TYPES,
    SUPPORTED_TRANSPORTS,
    MCPServerAuthConfig,
    MCPServerConfig,
    delete_mcp_server_config,
    list_mcp_server_configs,
    upsert_mcp_server_config,
)

router = APIRouter(prefix="/api", tags=["config"])
_MCP_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class MCPServerAuthResponse(BaseModel):
    type: str = "none"
    bearer_token_set: bool = False
    token_url: str | None = None
    client_id: str | None = None
    client_secret_set: bool = False
    scope: str | None = None
    audience: str | None = None
    grant_type: str = "client_credentials"
    refresh_skew_seconds: int = 30


class MCPServerConfigResponse(BaseModel):
    key: str
    transport: str
    url: str
    enabled: bool
    auth: MCPServerAuthResponse


class MCPServersConfigResponse(BaseModel):
    enable_mcp_tools: bool
    servers: list[MCPServerConfigResponse]


class ObservabilityLinkResponse(BaseModel):
    key: str
    label: str
    enabled: bool
    configured: bool
    url: str | None = None
    status: str
    details: str


class ObservabilityConfigResponse(BaseModel):
    links: list[ObservabilityLinkResponse]


class AppConfigResponse(BaseModel):
    region: str
    embed_model_id: str
    model_list: list[str]
    model_display_names: dict[str, str]
    collection_list: list[str]
    enable_user_feedback: bool
    observability: ObservabilityConfigResponse


class MCPServerAuthWrite(BaseModel):
    type: str = Field(default="none")
    bearer_token: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    scope: str | None = None
    audience: str | None = None
    grant_type: str = "client_credentials"
    refresh_skew_seconds: int = Field(default=30, gt=0)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_AUTH_TYPES:
            raise ValueError(
                "auth type must be one of: " + ", ".join(sorted(SUPPORTED_AUTH_TYPES))
            )
        return normalized

    @field_validator(
        "bearer_token",
        "token_url",
        "client_id",
        "client_secret",
        "scope",
        "audience",
        "grant_type",
        mode="before",
    )
    @classmethod
    def _strip_optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class MCPServerConfigWrite(BaseModel):
    transport: str = Field(default="streamable-http")
    url: str
    enabled: bool = True
    auth: MCPServerAuthWrite = Field(default_factory=MCPServerAuthWrite)

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


class MCPConnectionToolResponse(BaseModel):
    name: str
    description: str


class MCPConnectionTestResponse(BaseModel):
    key: str
    ok: bool
    tool_count: int
    tools: list[MCPConnectionToolResponse]
    error: str | None


def _is_placeholder_value(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return not normalized or normalized.startswith(("pk-lf-your-", "sk-lf-your-"))


def _observability_links(settings: Settings) -> ObservabilityConfigResponse:
    langfuse_enabled = bool(getattr(settings, "ENABLE_LANGFUSE_TRACING", False))
    langfuse_host = (getattr(settings, "LANGFUSE_HOST", "") or "").strip() or None
    langfuse_ui_url = (
        getattr(settings, "LANGFUSE_UI_URL", None) or langfuse_host or ""
    ).strip() or None
    langfuse_has_keys = not _is_placeholder_value(
        getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    ) and not _is_placeholder_value(getattr(settings, "LANGFUSE_SECRET_KEY", None))
    langfuse_configured = langfuse_enabled and bool(langfuse_host) and langfuse_has_keys

    grafana_enabled = bool(getattr(settings, "ENABLE_OBSERVABILITY_STACK", False))
    grafana_url = (getattr(settings, "GRAFANA_URL", "") or "").strip() or None

    logging_enabled = bool(getattr(settings, "ENABLE_OCI_LOGGING_ANALYTICS", False))
    logging_namespace = (getattr(settings, "LOGGING_ANALYTICS_NAMESPACE", None) or "").strip()
    logging_group = (getattr(settings, "LOGGING_ANALYTICS_LOG_GROUP_ID", None) or "").strip()
    logging_url = (
        getattr(settings, "LOGGING_ANALYTICS_CONSOLE_URL", None) or ""
    ).strip() or None
    logging_configured = logging_enabled and bool(logging_namespace and logging_group)

    return ObservabilityConfigResponse(
        links=[
            ObservabilityLinkResponse(
                key="langfuse",
                label="Langfuse",
                enabled=langfuse_enabled,
                configured=langfuse_configured,
                url=langfuse_ui_url if langfuse_configured else None,
                status="Ready" if langfuse_configured else "Disabled" if not langfuse_enabled else "Needs keys",
                details=(
                    "Tracing is enabled and the Langfuse host is configured."
                    if langfuse_configured
                    else "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
                    if langfuse_enabled
                    else "Set ENABLE_LANGFUSE_TRACING=true to send traces."
                ),
            ),
            ObservabilityLinkResponse(
                key="grafana",
                label="Grafana",
                enabled=grafana_enabled,
                configured=grafana_enabled and bool(grafana_url),
                url=grafana_url if grafana_enabled and grafana_url else None,
                status="Ready" if grafana_enabled and grafana_url else "Disabled",
                details=(
                    "Local observability stack is enabled."
                    if grafana_enabled
                    else "Set ENABLE_OBSERVABILITY_STACK=true to run local Grafana."
                ),
            ),
            ObservabilityLinkResponse(
                key="oracle_logging_analytics",
                label="Oracle Logging Analytics",
                enabled=logging_enabled,
                configured=logging_configured,
                url=logging_url if logging_configured else None,
                status=(
                    "Ready"
                    if logging_configured and logging_url
                    else "Configured"
                    if logging_configured
                    else "Disabled"
                    if not logging_enabled
                    else "Needs config"
                ),
                details=(
                    "Logging Analytics is configured. Add LOGGING_ANALYTICS_CONSOLE_URL for a direct link."
                    if logging_configured and not logging_url
                    else "Logs are configured for OCI Logging Analytics."
                    if logging_configured
                    else "Set namespace and log group in .env."
                    if logging_enabled
                    else "Set ENABLE_OCI_LOGGING_ANALYTICS=true to send logs."
                ),
            ),
        ]
    )


@router.get("/config", response_model=AppConfigResponse)
async def get_config(request: Request) -> AppConfigResponse:
    settings: Settings = get_settings_dep(request)
    return AppConfigResponse(
        region=settings.REGION,
        embed_model_id=settings.EMBED_MODEL_ID,
        model_list=settings.MODEL_LIST,
        model_display_names=settings.MODEL_DISPLAY_NAMES,
        collection_list=settings.COLLECTION_LIST or [settings.DEFAULT_COLLECTION],
        enable_user_feedback=settings.ENABLE_USER_FEEDBACK,
        observability=_observability_links(settings),
    )


def _settings_mcp_config(settings: Settings) -> dict[str, dict[str, object]]:
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


def _to_auth_response(auth: MCPServerAuthConfig) -> MCPServerAuthResponse:
    return MCPServerAuthResponse(
        type=auth.type,
        bearer_token_set=bool(auth.bearer_token),
        token_url=auth.token_url,
        client_id=auth.client_id,
        client_secret_set=bool(auth.client_secret),
        scope=auth.scope,
        audience=auth.audience,
        grant_type=auth.grant_type,
        refresh_skew_seconds=auth.refresh_skew_seconds,
    )


def _to_response(server: MCPServerConfig) -> MCPServerConfigResponse:
    return MCPServerConfigResponse(
        key=server.key,
        transport=server.transport,
        url=server.url,
        enabled=server.enabled,
        auth=_to_auth_response(server.auth),
    )


def _auth_from_write(
    payload: MCPServerAuthWrite,
    *,
    existing: MCPServerAuthConfig | None,
) -> MCPServerAuthConfig:
    if payload.type == "bearer":
        bearer_token = payload.bearer_token
        if not bearer_token and existing and existing.type == "bearer":
            bearer_token = existing.bearer_token
        return MCPServerAuthConfig(type="bearer", bearer_token=bearer_token)

    if payload.type == "oauth_client_credentials":
        client_secret = payload.client_secret
        if not client_secret and existing and existing.type == "oauth_client_credentials":
            client_secret = existing.client_secret
        return MCPServerAuthConfig(
            type="oauth_client_credentials",
            token_url=payload.token_url,
            client_id=payload.client_id,
            client_secret=client_secret,
            scope=payload.scope,
            audience=payload.audience,
            grant_type=payload.grant_type or "client_credentials",
            refresh_skew_seconds=payload.refresh_skew_seconds,
        )

    return MCPServerAuthConfig()


def _runtime_auth_config(auth: MCPServerAuthConfig) -> dict[str, object] | None:
    if auth.type == "bearer":
        return {"type": "bearer", "bearer_token": auth.bearer_token}
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
    return None


def _server_runtime_config(server: MCPServerConfig) -> dict[str, object]:
    config: dict[str, object] = {
        "transport": server.transport,
        "url": server.url,
    }
    auth = _runtime_auth_config(server.auth)
    if auth is not None:
        config["auth"] = auth
    return config


def _current_server(settings: Settings, key: str) -> MCPServerConfig | None:
    servers = list_mcp_server_configs(
        base_config=_settings_mcp_config(settings),
        store_path=_store_path(settings),
    )
    return next((server for server in servers if server.key == key), None)


def _server_config_for_test(
    *,
    key: str,
    payload: MCPServerConfigWrite | None,
    settings: Settings,
) -> MCPServerConfig:
    current = _current_server(settings, key)
    if payload is not None:
        return MCPServerConfig(
            key=key,
            transport=payload.transport,
            url=payload.url,
            enabled=True,
            auth=_auth_from_write(payload.auth, existing=current.auth if current else None),
        )

    if current is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return MCPServerConfig(
        key=current.key,
        transport=current.transport,
        url=current.url,
        enabled=True,
        auth=current.auth,
    )


def _test_run_config(server: MCPServerConfig) -> dict[str, object]:
    return {
        "configurable": {
            "mcp_servers_config_override": {
                server.key: _server_runtime_config(server)
            }
        }
    }


def _connection_test_timeout(settings: Settings) -> float:
    raw_value = getattr(settings, "MCP_CONNECTION_TEST_TIMEOUT_SECONDS", 8.0)
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError):
        return 8.0
    return timeout if timeout > 0 else 8.0


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
    normalized_key = _validate_mcp_key(key)
    current = _current_server(settings, normalized_key)
    server = upsert_mcp_server_config(
        MCPServerConfig(
            key=normalized_key,
            transport=payload.transport,
            url=payload.url,
            enabled=payload.enabled,
            auth=_auth_from_write(payload.auth, existing=current.auth if current else None),
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
        auth=current.auth,
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


@router.post(
    "/config/mcp-servers/{key}/test",
    response_model=MCPConnectionTestResponse,
)
async def test_mcp_server_connection_api(
    key: str,
    request: Request,
    payload: MCPServerConfigWrite | None = None,
) -> MCPConnectionTestResponse:
    settings: Settings = get_settings_dep(request)
    normalized_key = _validate_mcp_key(key)
    server = _server_config_for_test(
        key=normalized_key,
        payload=payload,
        settings=settings,
    )
    timeout_seconds = _connection_test_timeout(settings)
    try:
        tools = await asyncio.wait_for(
            mcp_adapter_runtime.load_adapter_tools(
                server_keys=[server.key],
                run_config=_test_run_config(server),
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return MCPConnectionTestResponse(
            key=server.key,
            ok=False,
            tool_count=0,
            tools=[],
            error=f"Connection test timed out after {timeout_seconds:g} seconds",
        )
    except Exception as exc:  # noqa: BLE001
        return MCPConnectionTestResponse(
            key=server.key,
            ok=False,
            tool_count=0,
            tools=[],
            error=str(exc) or type(exc).__name__,
        )

    tool_summaries = [
        MCPConnectionToolResponse(
            name=str(getattr(tool, "name", "") or "").strip(),
            description=str(getattr(tool, "description", "") or "").strip(),
        )
        for tool in tools
        if str(getattr(tool, "name", "") or "").strip()
    ]
    return MCPConnectionTestResponse(
        key=server.key,
        ok=True,
        tool_count=len(tool_summaries),
        tools=tool_summaries,
        error=None,
    )

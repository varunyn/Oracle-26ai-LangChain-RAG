"""
Pydantic Settings for the RAG API.

- Canonical location: api/settings.py
- Precedence: environment variables > .env file > built-in defaults
- Copy .env.example to .env and set values; .env is gitignored.
- See docs/CONFIGURATION.md for all options.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import ClassVar

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import override


class Settings(BaseSettings):
    # Env var names match .env.example (no prefix, case-sensitive)
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # =============================================================================
    # GENERAL
    # =============================================================================
    DEBUG: bool = False
    AUTH: str = "API_KEY"  # "API_KEY" | "SECURITY_TOKEN"
    OCI_PROFILE: str = "CHICAGO"
    OCI_CONFIG_FILE: str = "~/.oci/config"

    # =============================================================================
    # OCI GEN AI
    # =============================================================================
    REGION: str = "us-chicago-1"
    COMPARTMENT_ID: str = "ocid1.compartment.oc1..your-compartment-ocid"
    SERVICE_ENDPOINT: str | None = None  # computed from REGION if None

    # =============================================================================
    # LLM
    # =============================================================================
    LLM_MODEL_ID: str = "meta.llama-3.3-70b-instruct"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4000

    # Union so dotenv can pass string; validators normalize. Empty => model_validator fills from REGION.
    MODEL_LIST: str | list[str] = Field(default_factory=list)
    MODEL_DISPLAY_NAMES: str | dict[str, str] = Field(default_factory=dict)

    @field_validator("MODEL_LIST", mode="before")
    @classmethod
    def _parse_model_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                out = json.loads(s)
                if isinstance(out, list):
                    return [str(x).strip() for x in out if str(x).strip()]
            except Exception:
                pass
        return [k.strip() for k in s.split(",") if k.strip()]

    @field_validator("MODEL_DISPLAY_NAMES", mode="before")
    @classmethod
    def _parse_model_display_names(cls, v: object) -> dict[str, str]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items() if str(k).strip()}
        s = str(v).strip()
        if not s or not s.startswith("{"):
            return {}
        try:
            out = json.loads(s)
            if isinstance(out, dict):
                return {str(k): str(val) for k, val in out.items() if str(k).strip()}
        except Exception:
            pass
        return {}

    # =============================================================================
    # EMBEDDINGS
    # =============================================================================
    EMBED_MODEL_TYPE: str = "OCI"
    EMBED_MODEL_ID: str = "cohere.embed-v4.0"

    # =============================================================================
    # ORACLE VECTOR STORE
    # =============================================================================
    VECTOR_DB_USER: str | None = None
    VECTOR_DB_PWD: str | None = None
    VECTOR_DSN: str | None = None
    VECTOR_WALLET_DIR: str | None = None
    VECTOR_WALLET_PWD: str | None = None

    CONNECT_ARGS: dict[str, str | None] | None = None
    DB_TCP_CONNECT_TIMEOUT: int = 5

    # =============================================================================
    # RAG / SEARCH
    # =============================================================================
    RAG_SEARCH_MODE: str = "vector"
    # Union so dotenv can pass string; validator normalizes to list[str]
    COLLECTION_LIST: str | list[str] = Field(default_factory=lambda: ["RAG_KNOWLEDGE_BASE"])

    @field_validator("COLLECTION_LIST", mode="before")
    @classmethod
    def _parse_collection_list(cls, v: object) -> list[str]:
        if v is None:
            return ["RAG_KNOWLEDGE_BASE"]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()] or ["RAG_KNOWLEDGE_BASE"]
        s = str(v).strip()
        if not s:
            return ["RAG_KNOWLEDGE_BASE"]
        if s.startswith("["):
            try:
                out = json.loads(s)
                if isinstance(out, list):
                    return [str(x).strip() for x in out if str(x).strip()] or ["RAG_KNOWLEDGE_BASE"]
            except Exception:
                pass
        return [k.strip() for k in s.split(",") if k.strip()] or ["RAG_KNOWLEDGE_BASE"]

    DEFAULT_COLLECTION: str = "RAG_KNOWLEDGE_BASE"
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 100
    ENABLE_RERANKER: bool = True

    # =============================================================================
    # UI
    # =============================================================================
    ENABLE_USER_FEEDBACK: bool = True
    ENABLE_CORS: bool = True
    CORS_ALLOW_ORIGINS: str | list[str] = Field(
        default_factory=lambda: [
            "http://localhost:4000",
            "http://127.0.0.1:4000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_allow_origins(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                out = json.loads(s)
                if isinstance(out, list):
                    return [str(x).strip() for x in out if str(x).strip()]
            except Exception:
                pass
        return [k.strip() for k in s.split(",") if k.strip()]

    # =============================================================================
    # MCP (client)
    # =============================================================================
    ENABLE_MCP_TOOLS: bool = True
    MCP_SERVER_KEYS: list[str] | None = None  # .env: comma-separated e.g. "default,context7"

    @field_validator("MCP_SERVER_KEYS", mode="before")
    @classmethod
    def _parse_mcp_server_keys(cls, v: object) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()] or None
        s = str(v).strip()
        if not s:
            return None
        return [k.strip() for k in s.split(",") if k.strip()] or None

    # 0 = bind every loaded MCP tool (slowest with many tools). Set e.g. 12 + ALWAYS_INCLUDE for faster turns.
    MCP_TOOL_SELECTION_MAX_TOOLS: int = 0
    MCP_TOOL_SELECTION_ALWAYS_INCLUDE: list[str] = Field(default_factory=list)

    @field_validator("MCP_TOOL_SELECTION_ALWAYS_INCLUDE", mode="before")
    @classmethod
    def _parse_mcp_tool_selection_always_include(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if not s:
            return []
        return [k.strip() for k in s.split(",") if k.strip()]

    MCP_SEARCH_MODE: str = "vector"
    ENABLE_MCP_CLIENT_JWT: bool = False

    @field_validator("RAG_SEARCH_MODE", "MCP_SEARCH_MODE", mode="before")
    @classmethod
    def _parse_search_mode(cls, v: object) -> str:
        allowed = {"vector", "hybrid", "text"}
        mode = str(v or "vector").strip().lower()
        return mode if mode in allowed else "vector"

    MCP_SERVERS_CONFIG: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"},
            "context7": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"},
            "calculator": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"},
        }
    )

    @field_validator("MCP_SERVERS_CONFIG", mode="before")
    @classmethod
    def _parse_mcp_servers_config(cls, v: object) -> dict[str, dict[str, str]]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): dict(val) for k, val in v.items() if isinstance(val, dict)}
        s = str(v).strip()
        if not s:
            return {}
        if s.startswith("{"):
            try:
                out = json.loads(s)
                if isinstance(out, dict):
                    return {str(k): dict(val) for k, val in out.items() if isinstance(val, dict)}
            except Exception:
                pass
        return {}

    TRANSPORT: str = "streamable-http"  # "streamable-http" | "stdio"
    HOST: str = "0.0.0.0"
    PORT: int = 9000

    # =============================================================================
    # APM / OpenTelemetry (optional)
    # =============================================================================
    ENABLE_OTEL_TRACING: bool = False
    OTEL_TRACES_ENDPOINT: str | None = None
    OTEL_TRACES_HEADERS: dict[str, str] | None = None
    OTEL_LOGS_ENDPOINT: str | None = None
    ENABLE_OBSERVABILITY_STACK: bool = False

    ENABLE_LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: str = "http://localhost:3300"
    LANGFUSE_PUBLIC_KEY: str = "pk-lf-your-project-key"
    LANGFUSE_SECRET_KEY: str = "sk-lf-your-secret-key"
    LANGFUSE_TRACING_ENVIRONMENT: str = "development"
    LANGFUSE_ENVIRONMENT: str | None = None
    LANGFUSE_RELEASE: str | None = None

    # =============================================================================
    # Local Docker stacks (optional helper)
    # =============================================================================
    DOCKER_STACKS: dict[str, dict[str, object]] = Field(
        default_factory=lambda: {
            "core": {
                "enabled": False,
                "compose_file": "docker-compose.yml",
                "services": ["backend", "frontend"],
                "profiles": [],
            },
            "observability": {
                "enabled": False,
                "compose_file": "docker-compose.yml",
                "services": ["loki", "tempo", "otel-collector", "grafana"],
                "profiles": ["observability"],
            },
            "langfuse": {
                "enabled": False,
                "compose_file": "observability/langfuse/docker-compose.yml",
                "services": [],
                "profiles": [],
                "env_file": "observability/langfuse/.env",
            },
        }
    )

    # =============================================================================
    # OCI Logging Analytics (optional)
    # =============================================================================
    ENABLE_OCI_LOGGING_ANALYTICS: bool = False
    LOGGING_ANALYTICS_NAMESPACE: str | None = None
    LOGGING_ANALYTICS_LOG_GROUP_ID: str | None = None
    LOGGING_ANALYTICS_LOG_SET: str | None = None
    LOGGING_ANALYTICS_RESOURCE_CATEGORY: str = "rag-api"
    LOGGING_ANALYTICS_META_PROPERTIES: str | None = None

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order: env vars > .env file > class defaults
        return (env_settings, dotenv_settings, init_settings)

    @model_validator(mode="after")
    def _compute_defaults(self) -> Settings:
        # SERVICE_ENDPOINT defaults to region-specific endpoint if not explicitly set
        if not self.SERVICE_ENDPOINT and self.REGION:
            object.__setattr__(
                self,
                "SERVICE_ENDPOINT",
                f"https://inference.generativeai.{self.REGION}.oci.oraclecloud.com",
            )

        # MODEL_LIST / MODEL_DISPLAY_NAMES defaults depend on REGION when not provided
        if not self.MODEL_LIST:
            if self.REGION == "us-chicago-1":
                object.__setattr__(
                    self,
                    "MODEL_LIST",
                    [
                        "xai.grok-3",
                        "xai.grok-4",
                        "openai.gpt-4.1",
                        "openai.gpt-4o",
                        "openai.gpt-5",
                        "meta.llama-3.3-70b-instruct",
                        "cohere.command-a-03-2025",
                    ],
                )
                object.__setattr__(
                    self,
                    "MODEL_DISPLAY_NAMES",
                    {
                        "xai.grok-3": "Grok 3",
                        "xai.grok-4": "Grok 4",
                        "openai.gpt-4.1": "GPT-4.1",
                        "openai.gpt-4o": "GPT-4o",
                        "openai.gpt-5": "GPT-5",
                        "meta.llama-3.3-70b-instruct": "Llama 3.3 70B",
                        "cohere.command-a-03-2025": "Cohere Command A 03 2025",
                    },
                )
            else:
                object.__setattr__(
                    self,
                    "MODEL_LIST",
                    [
                        "meta.llama-3.3-70b-instruct",
                        "cohere.command-a-03-2025",
                        "openai.gpt-4.1",
                        "openai.gpt-4o",
                        "openai.gpt-5",
                    ],
                )
                object.__setattr__(
                    self,
                    "MODEL_DISPLAY_NAMES",
                    {
                        "meta.llama-3.3-70b-instruct": "Llama 3.3 70B",
                        "cohere.command-a-03-2025": "Cohere Command A 03 2025",
                        "openai.gpt-4.1": "GPT-4.1",
                        "openai.gpt-4o": "GPT-4o",
                        "openai.gpt-5": "GPT-5",
                    },
                )

        # If CONNECT_ARGS not explicitly provided, derive from individual VECTOR_* fields when available
        if self.CONNECT_ARGS is None and (
            self.VECTOR_DB_USER
            or self.VECTOR_DB_PWD
            or self.VECTOR_DSN
            or self.VECTOR_WALLET_DIR
            or self.VECTOR_WALLET_PWD
        ):
            object.__setattr__(
                self,
                "CONNECT_ARGS",
                {
                    "user": self.VECTOR_DB_USER,
                    "password": self.VECTOR_DB_PWD,
                    "dsn": self.VECTOR_DSN,
                    "config_dir": self.VECTOR_WALLET_DIR,
                    "wallet_location": self.VECTOR_WALLET_DIR,
                    "wallet_password": self.VECTOR_WALLET_PWD,
                },
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Configure via env vars or .env (see .env.example)."""
    return Settings()  # type: ignore[call-arg]

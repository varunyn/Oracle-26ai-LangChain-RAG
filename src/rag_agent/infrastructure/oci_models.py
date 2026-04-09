"""OCI GenAI LLM/embeddings via ChatOCIGenAI and OCIGenAIEmbeddings; auth from config or OCI_KEY_FILE."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings
from langchain_oracledb import OracleVS

from api.settings import get_settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_oci_config_path(raw_path: str) -> str:
    """Resolve OCI_CONFIG_FILE to an absolute path. Relative paths are relative to project root."""
    path = raw_path.strip()
    if not path:
        path = "~/.oci/config"
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        resolved = expanded
    else:
        resolved = (_PROJECT_ROOT / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"OCI config file not found at {resolved}. "
            "Create it (e.g. copy from ~/.oci/config to local-config/oci/config) or set "
            "OCI_CONFIG_FILE in .env (e.g. ~/.oci/config). "
            "See https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm"
        )
    return str(resolved)


def _write_config_with_key(config_path: Path, key_path: str) -> str | None:
    """Write a temp OCI config with key_file set to key_path. Returns temp path or None on failure."""
    text = config_path.read_text()
    new_text = re.sub(
        r"^\s*key_file\s*=.+$",
        f"key_file={key_path}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".oci.config", prefix="oci_")
    try:
        os.close(fd)
        Path(tmp_path).write_text(new_text)
        return tmp_path
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _resolve_oci_config_key_file(config_path: str) -> str:
    """Use OCI_KEY_FILE env if set and valid; else resolve relative key_file from config dir. Returns config path (original or temp)."""
    path = Path(config_path)
    if not path.is_file():
        return config_path

    env_key = os.environ.get("OCI_KEY_FILE", "").strip()
    if env_key and Path(env_key).expanduser().is_file():
        out = _write_config_with_key(path, env_key)
        return out or config_path

    match = re.search(r"^\s*key_file\s*=\s*(.+)$", path.read_text(), re.MULTILINE)
    if not match:
        return config_path
    value = match.group(1).strip().strip('"').strip("'")
    key_path = Path(value).expanduser()
    if key_path.is_absolute() and key_path.is_file():
        return config_path
    resolved = (path.resolve().parent / value).resolve()
    if not resolved.is_file():
        return config_path
    out = _write_config_with_key(path, str(resolved))
    return out or config_path


def _get_oci_auth_file_location() -> str:
    """Resolve OCI config path and key_file for LLM and embeddings."""
    raw = getattr(get_settings(), "OCI_CONFIG_FILE", "~/.oci/config")
    path = _resolve_oci_config_path(raw)
    return _resolve_oci_config_key_file(path)


ALLOWED_EMBED_MODELS_TYPE = {"OCI"}

_llm_cache: dict = {}
_embed_cache: dict = {}

MODELS_WITHOUT_KWARGS = {
    "openai.gpt-5",
    "openai.gpt-4o-search-preview",
    "openai.gpt-4o-search-preview-2025-03-11",
}


def _normalize_provider(model_id: str) -> str:
    """Map model_id prefix to ChatOCIGenAI provider.

    - Llama: meta.* -> "meta" (also xai/openai routed to meta).
    - Gemini: google.* -> "generic".
    - Others (e.g. cohere) passed through as-is.
    """
    provider = (model_id or "").split(".")[0].lower()
    if provider in {"xai", "openai"}:
        return "meta"
    if provider == "google":
        return "generic"
    return provider


def get_llm(
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOCIGenAI:
    """
    Return a ChatOCIGenAI instance for OCI Gen AI (same auth pattern as main).
    Used for RAG and MCP. Auth from config: OCI_PROFILE, OCI_CONFIG_FILE (local-config).
    Clients are cached by (model_id, temperature, max_tokens).
    """
    if model_id is None:
        model_id = get_settings().LLM_MODEL_ID
    if temperature is None:
        temperature = get_settings().TEMPERATURE
    if max_tokens is None:
        max_tokens = get_settings().MAX_TOKENS

    cache_key = (model_id, float(temperature), max_tokens)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    auth_file_location = _get_oci_auth_file_location()
    profile = get_settings().OCI_PROFILE or "DEFAULT"
    provider = _normalize_provider(model_id)

    endpoint = (
        getattr(get_settings(), "SERVICE_ENDPOINT", None)
        or f"https://inference.generativeai.{get_settings().REGION}.oci.oraclecloud.com"
    ).rstrip("/")

    llm_kwargs = {
        "auth_type": get_settings().AUTH,
        "model_id": model_id,
        "service_endpoint": endpoint,
        "compartment_id": get_settings().COMPARTMENT_ID,
        "is_stream": True,
        "provider": provider,
    }
    if profile:
        llm_kwargs["auth_profile"] = profile
        llm_kwargs["auth_file_location"] = auth_file_location
        logger.info("Using OCI profile: %s from %s", profile, auth_file_location)
    if model_id not in MODELS_WITHOUT_KWARGS:
        model_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Do not add tool_result_guidance: OCI GenAI rejects it for non-tool
        # invocations (e.g. DirectAnswer) and some providers (e.g. Grok/xai).
        # Use only temperature and max_tokens.
        llm_kwargs["model_kwargs"] = model_kwargs
    else:
        llm_kwargs["model_kwargs"] = None

    llm = ChatOCIGenAI(**llm_kwargs)
    _llm_cache[cache_key] = llm
    logger.info("OCI Gen AI via ChatOCIGenAI (profile=%s, model=%s)", profile, model_id)
    return llm


def get_embedding_model(model_type: str = "OCI") -> OCIGenAIEmbeddings:
    """
    Initialize and return an instance of OCIGenAIEmbeddings.
    The embedding model is cached by model_type so repeated calls reuse the same instance.

    Returns:
        OCIGenAIEmbeddings: An instance of the OCI GenAI embeddings model.
    """
    if model_type not in ALLOWED_EMBED_MODELS_TYPE:
        raise ValueError(
            f"Invalid value for model_type: must be one of {ALLOWED_EMBED_MODELS_TYPE}"
        )

    if model_type in _embed_cache:
        return _embed_cache[model_type]

    endpoint = (
        getattr(get_settings(), "SERVICE_ENDPOINT", None)
        or f"https://inference.generativeai.{get_settings().REGION}.oci.oraclecloud.com"
    ).rstrip("/")

    embed_kwargs = {
        "auth_type": get_settings().AUTH,
        "model_id": get_settings().EMBED_MODEL_ID,
        "service_endpoint": endpoint,
        "compartment_id": get_settings().COMPARTMENT_ID,
    }
    if get_settings().OCI_PROFILE:
        embed_kwargs["auth_profile"] = get_settings().OCI_PROFILE
        embed_kwargs["auth_file_location"] = _get_oci_auth_file_location()
        logger.info(
            "Using OCI profile: %s for embeddings from %s",
            get_settings().OCI_PROFILE,
            embed_kwargs["auth_file_location"],
        )

    embed_model = OCIGenAIEmbeddings(**embed_kwargs)
    logger.info("Embedding model is: %s", get_settings().EMBED_MODEL_ID)
    _embed_cache[model_type] = embed_model
    return embed_model


def get_oracle_vs(
    conn: oracledb.Connection,
    collection_name: str,
    embed_model: OCIGenAIEmbeddings,
) -> OracleVS:
    """
    Initialize and return an instance of OracleVS for vector search.

    OracleVS is not cached: it is bound to a connection at construction, so
    caching would require dedicating connections (undermining pooling). Per-request
    connection cost is already removed by using the connection pool; the embedding
    model is cached separately by embed_model_type.

    Args:
        conn: The database connection object.
        collection_name (str): The name of the collection (DB table) to search in.
        embed_model: The embedding model to use for vector search.
    """
    oracle_vs = OracleVS(
        client=conn,
        table_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        embedding_function=embed_model,
    )

    return oracle_vs

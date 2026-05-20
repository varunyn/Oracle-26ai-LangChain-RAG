"""Thin langchain_oci constructors with shared OCI auth/config resolution."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings
from langchain_oci.common.auth import create_oci_client_kwargs
from langchain_oracledb import OracleVS
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference import models as oci_genai_models
from opentelemetry import trace

from api.settings import get_settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _uses_auth_profile(auth_type: str | None) -> bool:
    """True when auth requires config/profile-backed credentials."""
    normalized = str(auth_type or "").strip().upper()
    return normalized in {"API_KEY", "SECURITY_TOKEN"}


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


def _get_service_endpoint() -> str:
    return (
        getattr(get_settings(), "SERVICE_ENDPOINT", None)
        or f"https://inference.generativeai.{get_settings().REGION}.oci.oraclecloud.com"
    ).rstrip("/")


def get_llm(
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOCIGenAI:
    """Return a ChatOCIGenAI configured from shared OCI settings."""
    if model_id is None:
        model_id = get_settings().LLM_MODEL_ID
    if temperature is None:
        temperature = get_settings().TEMPERATURE
    if max_tokens is None:
        max_tokens = get_settings().MAX_TOKENS
    auth_type = get_settings().AUTH

    profile = get_settings().OCI_PROFILE or "DEFAULT"
    use_profile = _uses_auth_profile(auth_type)
    auth_file_location = _get_oci_auth_file_location() if use_profile else None

    endpoint = _get_service_endpoint()

    model_kw: dict[str, Any] = {"temperature": temperature}
    token_key = "max_completion_tokens" if model_id.startswith("openai.") else "max_tokens"
    model_kw[token_key] = max_tokens
    llm_kwargs: dict[str, Any] = {
        "auth_type": auth_type,
        "model_id": model_id,
        "service_endpoint": endpoint,
        "compartment_id": get_settings().COMPARTMENT_ID,
        "model_kwargs": model_kw,
        "max_sequential_tool_calls": getattr(get_settings(), "OCI_MAX_SEQUENTIAL_TOOL_CALLS", 10),
        "tool_result_guidance": bool(getattr(get_settings(), "OCI_TOOL_RESULT_GUIDANCE", True)),
    }
    if use_profile and profile and auth_file_location:
        llm_kwargs["auth_profile"] = profile
        llm_kwargs["auth_file_location"] = auth_file_location
        logger.info("Using OCI profile: %s from %s", profile, auth_file_location)

    llm = ChatOCIGenAI(**llm_kwargs)
    logger.info("OCI Gen AI via ChatOCIGenAI (profile=%s, model=%s)", profile, model_id)
    return llm


def get_embedding_model(model_type: str = "OCI") -> OCIGenAIEmbeddings:
    """Return an OCIGenAIEmbeddings instance configured from shared OCI settings."""
    if model_type != "OCI":
        raise ValueError("Invalid value for model_type: must be 'OCI'")

    endpoint = _get_service_endpoint()
    auth_type = get_settings().AUTH

    embed_kwargs = {
        "auth_type": auth_type,
        "model_id": get_settings().EMBED_MODEL_ID,
        "service_endpoint": endpoint,
        "compartment_id": get_settings().COMPARTMENT_ID,
    }
    if _uses_auth_profile(auth_type) and get_settings().OCI_PROFILE:
        embed_kwargs["auth_profile"] = get_settings().OCI_PROFILE
        embed_kwargs["auth_file_location"] = _get_oci_auth_file_location()
        logger.info(
            "Using OCI profile: %s for embeddings from %s",
            get_settings().OCI_PROFILE,
            embed_kwargs["auth_file_location"],
        )

    embed_model = OCIGenAIEmbeddings(**embed_kwargs)
    logger.info("Embedding model is: %s", get_settings().EMBED_MODEL_ID)
    return embed_model


def _get_rerank_serving_mode() -> object:
    endpoint_id = str(getattr(get_settings(), "RERANK_DEDICATED_ENDPOINT_ID", "") or "").strip()
    if endpoint_id:
        return oci_genai_models.DedicatedServingMode(endpoint_id=endpoint_id)
    return oci_genai_models.OnDemandServingMode(
        model_id=getattr(get_settings(), "RERANK_MODEL_ID", "cohere.rerank-v4.0-fast")
    )


def _get_rerank_client() -> GenerativeAiInferenceClient:
    settings = get_settings()
    auth_type = settings.AUTH
    profile = settings.OCI_PROFILE or "DEFAULT"
    auth_file_location = (
        _get_oci_auth_file_location() if _uses_auth_profile(auth_type) else "~/.oci/config"
    )
    client_kwargs = create_oci_client_kwargs(
        auth_type=auth_type,
        service_endpoint=_get_service_endpoint(),
        auth_file_location=auth_file_location,
        auth_profile=profile,
    )
    return GenerativeAiInferenceClient(**client_kwargs)


def rerank_documents(
    query: str,
    docs: list[Document],
    *,
    top_n: int | None = None,
) -> list[Document]:
    """Rerank LangChain documents with OCI Generative AI Inference rerankText."""
    if not query.strip() or not docs:
        return docs

    settings = get_settings()
    limit = max(1, min(int(top_n or getattr(settings, "RERANK_TOP_N", 5)), len(docs)))
    endpoint_id = str(getattr(settings, "RERANK_DEDICATED_ENDPOINT_ID", "") or "").strip()
    serving_mode_name = "dedicated" if endpoint_id else "on_demand"
    attributes = {
        "rag.rerank.enabled": True,
        "rag.rerank.provider": "oci",
        "rag.rerank.model_id": str(getattr(settings, "RERANK_MODEL_ID", "unknown") or "unknown"),
        "rag.rerank.input_docs": len(docs),
        "rag.rerank.top_n": limit,
        "rag.rerank.query_length": len(query or ""),
        "oci.genai.operation": "rerank_text",
        "oci.genai.serving_mode": serving_mode_name,
    }
    details_kwargs: dict[str, object] = {
        "input": query,
        "compartment_id": settings.COMPARTMENT_ID,
        "serving_mode": _get_rerank_serving_mode(),
        "documents": [str(doc.page_content or "") for doc in docs],
        "top_n": limit,
        "is_echo": False,
    }
    max_chunks = getattr(settings, "RERANK_MAX_CHUNKS_PER_DOCUMENT", None)
    if max_chunks is not None:
        details_kwargs["max_chunks_per_document"] = int(max_chunks)
    max_tokens = getattr(settings, "RERANK_MAX_TOKENS_PER_DOCUMENT", None)
    if max_tokens is not None:
        details_kwargs["max_tokens_per_document"] = int(max_tokens)

    details = oci_genai_models.RerankTextDetails(**details_kwargs)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("oci.rerank_text") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

        response = _get_rerank_client().rerank_text(details)
        data = getattr(response, "data", None)
        ranks = getattr(data, "document_ranks", None) or []
        ordered: list[tuple[float, Document]] = []
        seen: set[int] = set()
        for rank in ranks:
            index = getattr(rank, "index", None)
            if not isinstance(index, int) or index < 0 or index >= len(docs) or index in seen:
                continue
            score = float(getattr(rank, "relevance_score", 0.0) or 0.0)
            ordered.append((score, docs[index]))
            seen.add(index)
        ordered.sort(key=lambda item: item[0], reverse=True)
        reranked = [doc for _, doc in ordered[:limit]]

        output_docs = len(reranked)
        span.set_attribute("rag.rerank.output_docs", output_docs)
        log_attributes = {"event_type": "oci_rerank", **attributes, "rag.rerank.output_docs": output_docs}
        logger.info(
            "oci_rerank_success model_id=%s input_docs=%d output_docs=%d top_n=%d",
            log_attributes["rag.rerank.model_id"],
            len(docs),
            output_docs,
            limit,
            extra={"otel_attributes": log_attributes},
        )
        return reranked


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

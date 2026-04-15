from __future__ import annotations

from types import SimpleNamespace

from src.rag_agent.infrastructure import oci_models


def test_get_llm_builds_chat_oci_genai_with_shared_auth_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOCIGenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        AUTH="API_KEY",
        LLM_MODEL_ID="google.gemini-2.5-pro",
        TEMPERATURE=0.2,
        MAX_TOKENS=2048,
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(oci_models, "ChatOCIGenAI", FakeChatOCIGenAI)

    oci_models.get_llm()

    assert captured == {
        "auth_type": "API_KEY",
        "model_id": "google.gemini-2.5-pro",
        "service_endpoint": "https://example.oraclecloud.com",
        "compartment_id": "ocid1.compartment.oc1..example",
        "auth_profile": "CHICAGO",
        "auth_file_location": "/tmp/test.oci.config",
        "model_kwargs": {"temperature": 0.2, "max_tokens": 2048},
    }


def test_get_embedding_model_builds_oci_embeddings_with_shared_auth_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOCIGenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        AUTH="API_KEY",
        EMBED_MODEL_ID="cohere.embed-v4.0",
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(oci_models, "OCIGenAIEmbeddings", FakeOCIGenAIEmbeddings)

    oci_models.get_embedding_model()

    assert captured == {
        "auth_type": "API_KEY",
        "model_id": "cohere.embed-v4.0",
        "service_endpoint": "https://example.oraclecloud.com",
        "compartment_id": "ocid1.compartment.oc1..example",
        "auth_profile": "CHICAGO",
        "auth_file_location": "/tmp/test.oci.config",
    }


def test_get_llm_preserves_explicit_xai_model_id_without_provider_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOCIGenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        AUTH="API_KEY",
        LLM_MODEL_ID="xai.grok-4-1-fast-reasoning",
        TEMPERATURE=0.2,
        MAX_TOKENS=2048,
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(oci_models, "ChatOCIGenAI", FakeChatOCIGenAI)

    oci_models.get_llm()

    assert captured["model_id"] == "xai.grok-4-1-fast-reasoning"
    assert "provider" not in captured


def test_get_llm_uses_openai_max_completion_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOCIGenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        AUTH="API_KEY",
        LLM_MODEL_ID="openai.gpt-5",
        TEMPERATURE=0.2,
        MAX_TOKENS=2048,
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(oci_models, "ChatOCIGenAI", FakeChatOCIGenAI)

    oci_models.get_llm()

    assert captured["model_kwargs"] == {"temperature": 0.2, "max_completion_tokens": 2048}


def test_get_llm_does_not_pass_profile_for_instance_principal(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOCIGenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        AUTH="INSTANCE_PRINCIPAL",
        LLM_MODEL_ID="google.gemini-2.5-pro",
        TEMPERATURE=0.2,
        MAX_TOKENS=2048,
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "ChatOCIGenAI", FakeChatOCIGenAI)

    oci_models.get_llm()

    assert captured["auth_type"] == "INSTANCE_PRINCIPAL"
    assert "auth_profile" not in captured
    assert "auth_file_location" not in captured

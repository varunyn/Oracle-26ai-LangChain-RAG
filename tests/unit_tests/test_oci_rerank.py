from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from src.rag_agent.infrastructure import oci_models


def test_rerank_documents_uses_oci_rerank_text(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def rerank_text(self, details: object) -> object:
            captured["details"] = details
            return SimpleNamespace(
                data=SimpleNamespace(
                    document_ranks=[
                        SimpleNamespace(index=1, relevance_score=0.98),
                        SimpleNamespace(index=0, relevance_score=0.42),
                    ],
                ),
            )

    settings = SimpleNamespace(
        AUTH="API_KEY",
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
        RERANK_MODEL_ID="cohere.rerank-v4.0-fast",
        RERANK_DEDICATED_ENDPOINT_ID=None,
        RERANK_TOP_N=2,
        RERANK_MAX_CHUNKS_PER_DOCUMENT=None,
        RERANK_MAX_TOKENS_PER_DOCUMENT=None,
    )
    docs = [
        Document(page_content="A less relevant chunk", metadata={"source": "a"}),
        Document(page_content="The best matching chunk", metadata={"source": "b"}),
    ]

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(
        oci_models,
        "create_oci_client_kwargs",
        lambda **kwargs: {"config": {"region": "us-chicago-1"}, **kwargs},
    )
    monkeypatch.setattr(oci_models, "GenerativeAiInferenceClient", FakeClient)

    reranked = oci_models.rerank_documents("best chunk", docs)

    assert reranked == [docs[1], docs[0]]
    assert captured["client_kwargs"] == {
        "config": {"region": "us-chicago-1"},
        "auth_type": "API_KEY",
        "service_endpoint": "https://example.oraclecloud.com",
        "auth_file_location": "/tmp/test.oci.config",
        "auth_profile": "CHICAGO",
    }
    details = captured["details"]
    assert details.input == "best chunk"
    assert details.compartment_id == "ocid1.compartment.oc1..example"
    assert details.documents == ["A less relevant chunk", "The best matching chunk"]
    assert details.top_n == 2
    assert details.is_echo is False
    assert details.serving_mode.model_id == "cohere.rerank-v4.0-fast"


def test_rerank_documents_uses_dedicated_endpoint_when_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

        def rerank_text(self, details: object) -> object:
            captured["serving_mode"] = details.serving_mode
            return SimpleNamespace(data=SimpleNamespace(document_ranks=[]))

    settings = SimpleNamespace(
        AUTH="INSTANCE_PRINCIPAL",
        SERVICE_ENDPOINT=None,
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
        RERANK_MODEL_ID="cohere.rerank-v4.0-fast",
        RERANK_DEDICATED_ENDPOINT_ID="ocid1.generativeaiendpoint.oc1..example",
        RERANK_TOP_N=5,
        RERANK_MAX_CHUNKS_PER_DOCUMENT=None,
        RERANK_MAX_TOKENS_PER_DOCUMENT=None,
    )

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(
        oci_models,
        "create_oci_client_kwargs",
        lambda **kwargs: {"config": {}, **kwargs},
    )
    monkeypatch.setattr(oci_models, "GenerativeAiInferenceClient", FakeClient)

    reranked = oci_models.rerank_documents(
        "query",
        [Document(page_content="chunk", metadata={})],
    )

    assert reranked == []
    serving_mode = captured["serving_mode"]
    assert serving_mode.endpoint_id == "ocid1.generativeaiendpoint.oc1..example"


def test_rerank_documents_emits_otel_span_and_log_attributes(monkeypatch) -> None:
    captured: dict[str, object] = {"span_attributes": {}, "log_records": []}

    class FakeSpan:
        def __enter__(self) -> FakeSpan:
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def set_attribute(self, key: str, value: object) -> None:
            span_attributes = captured["span_attributes"]
            assert isinstance(span_attributes, dict)
            span_attributes[key] = value

    class FakeTracer:
        def start_as_current_span(self, name: str) -> FakeSpan:
            captured["span_name"] = name
            return FakeSpan()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

        def rerank_text(self, details: object) -> object:
            captured["details"] = details
            return SimpleNamespace(
                data=SimpleNamespace(
                    document_ranks=[SimpleNamespace(index=0, relevance_score=0.9)],
                ),
            )

    settings = SimpleNamespace(
        AUTH="API_KEY",
        SERVICE_ENDPOINT="https://example.oraclecloud.com",
        REGION="us-chicago-1",
        COMPARTMENT_ID="ocid1.compartment.oc1..example",
        OCI_PROFILE="CHICAGO",
        RERANK_MODEL_ID="cohere.rerank-v4.0-fast",
        RERANK_DEDICATED_ENDPOINT_ID=None,
        RERANK_TOP_N=1,
        RERANK_MAX_CHUNKS_PER_DOCUMENT=None,
        RERANK_MAX_TOKENS_PER_DOCUMENT=None,
    )

    def capture_log(*args: object, **kwargs: object) -> None:
        log_records = captured["log_records"]
        assert isinstance(log_records, list)
        log_records.append((args, kwargs))

    monkeypatch.setattr(oci_models, "get_settings", lambda: settings)
    monkeypatch.setattr(oci_models, "_get_oci_auth_file_location", lambda: "/tmp/test.oci.config")
    monkeypatch.setattr(
        oci_models,
        "create_oci_client_kwargs",
        lambda **kwargs: {"config": {"region": "us-chicago-1"}, **kwargs},
    )
    monkeypatch.setattr(oci_models, "GenerativeAiInferenceClient", FakeClient)
    monkeypatch.setattr(
        oci_models,
        "trace",
        SimpleNamespace(get_tracer=lambda name: FakeTracer()),
        raising=False,
    )
    monkeypatch.setattr(oci_models.logger, "info", capture_log)

    docs = [Document(page_content="The best matching chunk", metadata={"source": "a"})]
    reranked = oci_models.rerank_documents("best chunk", docs)

    assert reranked == docs
    assert captured["span_name"] == "oci.rerank_text"
    assert captured["span_attributes"] == {
        "rag.rerank.enabled": True,
        "rag.rerank.provider": "oci",
        "rag.rerank.model_id": "cohere.rerank-v4.0-fast",
        "rag.rerank.input_docs": 1,
        "rag.rerank.output_docs": 1,
        "rag.rerank.top_n": 1,
        "rag.rerank.query_length": 10,
        "oci.genai.operation": "rerank_text",
        "oci.genai.serving_mode": "on_demand",
    }
    log_records = captured["log_records"]
    assert isinstance(log_records, list)
    assert log_records
    _, kwargs = log_records[-1]
    assert kwargs["extra"] == {
        "otel_attributes": {
            "event_type": "oci_rerank",
            "rag.rerank.enabled": True,
            "rag.rerank.provider": "oci",
            "rag.rerank.model_id": "cohere.rerank-v4.0-fast",
            "rag.rerank.input_docs": 1,
            "rag.rerank.output_docs": 1,
            "rag.rerank.top_n": 1,
            "rag.rerank.query_length": 10,
            "oci.genai.operation": "rerank_text",
            "oci.genai.serving_mode": "on_demand",
        }
    }

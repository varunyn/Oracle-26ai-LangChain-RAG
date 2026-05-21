from types import SimpleNamespace

from scripts import create_rag_table


def test_create_vector_index_delegates_to_oraclevs_create_index(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOracleVS:
        def __init__(self, *, client, embedding_function, table_name, distance_strategy):
            captured["client"] = client
            captured["embedding_function"] = embedding_function
            captured["table_name"] = table_name
            captured["distance_strategy"] = distance_strategy

    def fake_create_index(client, vector_store, params):
        captured["index_client"] = client
        captured["vector_store"] = vector_store
        captured["params"] = params

    monkeypatch.setattr(create_rag_table, "OracleVS", FakeOracleVS)
    monkeypatch.setattr(create_rag_table, "create_index", fake_create_index)
    monkeypatch.setattr(create_rag_table, "_vector_index_exists", lambda conn, index_name: False)
    monkeypatch.setattr(create_rag_table, "get_embedding_model", lambda model_type: "embeddings")
    monkeypatch.setattr(
        create_rag_table,
        "get_settings",
        lambda: SimpleNamespace(EMBED_MODEL_TYPE="OCI"),
    )

    create_rag_table.create_vector_index(
        conn="connection",
        table_name="rag_knowledge_base",
        index_name="rag_hnsw_idx",
        index_type="HNSW",
    )

    assert captured["client"] == "connection"
    assert captured["embedding_function"] == "embeddings"
    assert captured["table_name"] == "RAG_KNOWLEDGE_BASE"
    assert captured["index_client"] == "connection"
    assert captured["params"] == {"idx_name": "RAG_HNSW_IDX", "idx_type": "HNSW"}


def test_create_vector_index_skips_existing_index(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fail_create_index(client, vector_store, params):
        captured["called"] = True

    monkeypatch.setattr(create_rag_table, "create_index", fail_create_index)
    monkeypatch.setattr(create_rag_table, "_vector_index_exists", lambda conn, index_name: True)

    create_rag_table.create_vector_index(
        conn="connection",
        table_name="rag_knowledge_base",
        index_name="rag_hnsw_idx",
    )

    assert captured == {}

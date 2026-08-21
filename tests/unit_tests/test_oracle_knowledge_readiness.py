import asyncio

from src.rag_agent.infrastructure.oracle_knowledge import KnowledgeReadinessProbe


class Settings:
    ORACLE_KNOWLEDGE_BASES = {"docs": "RAW"}
    ORACLE_KNOWLEDGE_DEFAULT_KEY = "docs"
    EMBED_MODEL_TYPE = "OCI"


def test_readiness_success_and_cache() -> None:
    calls = {"oracle": 0, "embedding": 0}
    probe = KnowledgeReadinessProbe(
        Settings(),
        oracle_check=lambda: calls.__setitem__("oracle", calls["oracle"] + 1) or True,
        embedding_check=lambda: calls.__setitem__("embedding", calls["embedding"] + 1) or True,
    )
    assert probe.check() == (True, "ready")
    assert probe.check() == (True, "ready")
    assert calls == {"oracle": 1, "embedding": 1}


def test_readiness_failures_short_circuit_and_are_safe() -> None:
    calls = []
    probe = KnowledgeReadinessProbe(
        Settings(),
        oracle_check=lambda: calls.append("oracle") or False,
        embedding_check=lambda: calls.append("embedding") or True,
    )
    assert probe.check() == (False, "oracle unavailable")
    assert calls == ["oracle"]
    bad = KnowledgeReadinessProbe(
        Settings(),
        oracle_check=lambda: (_ for _ in ()).throw(
            RuntimeError("password=secret dsn.internal:1521/ORCL SQL=SELECT secret")
        ),
        embedding_check=lambda: True,
    )
    assert bad.check() == (False, "readiness unavailable")


def test_async_readiness_offloads_blocking_probe() -> None:
    probe = KnowledgeReadinessProbe(
        Settings(), oracle_check=lambda: True, embedding_check=lambda: True
    )
    assert asyncio.run(probe.check_async()) == (True, "ready")


def test_default_provider_checks_use_expected_safe_boundaries(monkeypatch) -> None:
    import src.rag_agent.infrastructure.oracle_knowledge as module

    class Cursor:
        def __init__(self):
            self.sql = None
            self.closed = False

        def execute(self, sql):
            self.sql = sql

        def close(self):
            self.closed = True

    cursor = Cursor()

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def cursor(self):
            return cursor

    monkeypatch.setattr("src.rag_agent.infrastructure.db_utils.get_connection", lambda: Conn())
    monkeypatch.setattr(module, "get_connection", lambda: Conn())
    model = type(
        "Model", (), {"embed_query": lambda self, text: (assert_constant(text), [1.0])[1]}
    )()
    monkeypatch.setattr(module, "get_embedding_model", lambda _: model)
    probe = KnowledgeReadinessProbe(Settings())
    assert probe._check_oracle() is True and cursor.sql == "SELECT 1 FROM dual" and cursor.closed
    assert probe._check_embedding() is True


def assert_constant(value):
    assert value == "oracle knowledge readiness"

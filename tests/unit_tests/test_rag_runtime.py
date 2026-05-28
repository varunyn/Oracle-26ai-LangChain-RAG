from __future__ import annotations

from types import TracebackType

from src.rag_agent.runtime import rag_runtime


class FakeConnectionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def test_oracle_retrieval_tool_returns_error_content_when_vector_search_fails(
    monkeypatch,
) -> None:
    def failing_search_documents(**kwargs: object) -> list[object]:
        _ = kwargs
        raise RuntimeError("Failed due to a DB error: ORA-22275: invalid LOB locator specified")

    def fake_dependency(name: str, fallback: object) -> object:
        _ = fallback
        dependencies: dict[str, object] = {
            "get_pooled_connection": lambda: FakeConnectionContext(),
            "get_embedding_model": lambda: object(),
            "search_documents": failing_search_documents,
        }
        return dependencies[name]

    monkeypatch.setattr(rag_runtime, "_compat_dependency", fake_dependency)

    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="ORACLE_WEB_EMBEDDINGS",
        filter_docs=lambda query, docs: docs,
    )

    content = tool.invoke({"query": "Northway Solutions payment terms"})

    assert "Oracle retrieval failed" in content
    assert "ORA-22275" in content
    assert getattr(tool, "_retrieval_state") == {
        "docs": [],
        "error": "Failed due to a DB error: ORA-22275: invalid LOB locator specified",
    }

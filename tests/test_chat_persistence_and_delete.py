# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnusedParameter=false, reportUnusedImport=false, reportUnannotatedClassAttribute=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportPrivateUsage=false
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from httpx import ASGITransport
from langchain_core.messages import AnyMessage
from pytest import MonkeyPatch


@pytest.fixture()
def stub_llm(monkeypatch: MonkeyPatch) -> None:
    """Stub OCI LLM to avoid external calls (used by DirectAnswer)."""

    class _StubLLM:
        def invoke(
            self, _messages: list[object], config: object | None = None
        ) -> object:  # noqa: D401
            return SimpleNamespace(content="stubbed answer")

    def _get_llm_stub(*_a: object, **_k: object) -> object:
        return _StubLLM()

    monkeypatch.setattr(
        "src.rag_agent.infrastructure.oci_models.get_llm", _get_llm_stub, raising=True
    )


@pytest.fixture()
def temp_langgraph_sqlite(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Point LANGGRAPH_SQLITE_PATH to a temp DB and reset default checkpointer."""
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("LANGGRAPH_SQLITE_PATH", str(db_path))
    # Force langgraph.sqlite checkpointer to reinitialize using the new path
    monkeypatch.setattr(
        "src.rag_agent.langgraph.graph._DEFAULT_SQLITE_CHECKPOINTER", None, raising=False
    )


def test_last_message_must_be_user_returns_422() -> None:
    import asyncio

    async def run() -> None:
        from api.main import app

        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hi again"},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 422
            data = resp.json()
            assert isinstance(data.get("detail"), str)
            assert isinstance(data.get("errors"), list)

    asyncio.run(run())


def test_multi_message_history_is_accepted() -> None:
    import asyncio

    async def run() -> None:
        from api.main import app

        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": "Preserve user instructions."},
                {"role": "user", "content": "Use markdown if I ask for it."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "What did I ask you to do?"},
            ],
            "stream": False,
            "mode": "direct",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 200

    asyncio.run(run())


def test_persistence_across_two_calls_messages_length_is_4(
    temp_langgraph_sqlite: None, stub_llm: None
) -> None:
    import asyncio

    async def run() -> None:
        # Import app only after env + monkeypatches are applied
        from api.dependencies import build_chat_config
        from api.main import app

        thread_id = f"t_{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            headers = {"Content-Type": "application/json"}

            # Turn 1
            payload1 = {
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "thread_id": thread_id,
                "mode": "direct",
            }
            r1 = await client.post("/api/chat", headers=headers, json=payload1)
            assert r1.status_code == 200

            # Turn 2
            payload2 = {
                "messages": [{"role": "user", "content": "What did I just say?"}],
                "stream": False,
                "thread_id": thread_id,
                "mode": "direct",
            }
            r2 = await client.post("/api/chat", headers=headers, json=payload2)
            assert r2.status_code == 200

            from api.deps import request as deps
            from api.services.graph_service import GraphService

            # Use same graph_service as the app (async checkpointer); fallback for non-ASGI
            resources = getattr(app.state, "resources", None)
            graph_service = (
                resources.graph_service
                if resources
                else (deps._fallback_graph_service or GraphService())
            )
            run_config = build_chat_config(thread_id=thread_id, mode="direct")
            snapshot = await graph_service.get_state(run_config)
            vals = getattr(snapshot, "values", None) or {}
            msgs = cast(list[object], vals.get("messages") or [])
            assert len(msgs) == 4  # Human+AI per turn accumulated across 2 turns

    asyncio.run(run())


def test_delete_thread_clears_memory_and_is_idempotent(
    temp_langgraph_sqlite: None, stub_llm: None
) -> None:
    import asyncio

    async def run() -> None:
        # Import app only after env + monkeypatches are applied
        from api.dependencies import build_chat_config
        from api.main import app

        thread_id = f"t_{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            headers = {"Content-Type": "application/json"}

            # Create a 2-turn thread
            for content in ("Turn one", "Turn two"):
                payload = {
                    "messages": [{"role": "user", "content": content}],
                    "stream": False,
                    "thread_id": thread_id,
                    "mode": "direct",
                }
                resp = await client.post("/api/chat", headers=headers, json=payload)
                assert resp.status_code == 200

            # First delete → 204
            d1 = await client.delete(f"/api/threads/{thread_id}")
            assert d1.status_code == 204

            # Second delete → 404
            d2 = await client.delete(f"/api/threads/{thread_id}")
            assert d2.status_code == 404
            body_404 = cast(dict[str, object], d2.json())
            assert body_404.get("error") == "Thread not found"

            from api.deps import request as deps
            from api.services.graph_service import GraphService

            # Use same graph_service as the app (async checkpointer); fallback for non-ASGI
            resources = getattr(app.state, "resources", None)
            graph_service = (
                resources.graph_service
                if resources
                else (deps._fallback_graph_service or GraphService())
            )
            run_config = build_chat_config(thread_id=thread_id, mode="direct")
            snapshot = await graph_service.get_state(run_config)
            vals = getattr(snapshot, "values", None) or {}
            assert not vals, f"Expected empty state after delete, got: {vals}"

    asyncio.run(run())


def test_new_turn_resets_stale_standalone_question_before_search(
    temp_langgraph_sqlite: None,
) -> None:
    import asyncio

    async def run() -> None:
        from api.main import app
        from src.rag_agent.agent_state import State

        class RecordingGraphService:
            def __init__(self) -> None:
                self.captured_state: State | None = None

            def invoke(self, state: State, _run_config: dict[str, object]) -> State:
                self.captured_state = state
                messages = cast(Sequence[AnyMessage], state.get("messages") or [])
                return {
                    "user_request": str(state.get("user_request") or ""),
                    "messages": list(messages),
                    "final_answer": "ok",
                    "standalone_question": str(state.get("user_request") or ""),
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": None,
                    "mcp_used": False,
                    "mcp_tools_used": [],
                    "error": None,
                }

        recording_graph_service = RecordingGraphService()
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the required properties for a business object?",
                }
            ],
            "stream": False,
            "thread_id": f"t_{uuid.uuid4().hex[:8]}",
            "mode": "rag",
        }

        from api.dependencies import get_graph_service

        app.dependency_overrides[get_graph_service] = lambda: recording_graph_service
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                resp = await client.post("/api/chat", headers=headers, json=payload)
                assert resp.status_code == 200
        finally:
            app.dependency_overrides.clear()

        assert recording_graph_service.captured_state is not None
        captured_state = recording_graph_service.captured_state
        assert captured_state.get("user_request") == (
            "What are the required properties for a business object?"
        )
        assert captured_state.get("standalone_question") is None
        assert captured_state.get("history_text") is None

    asyncio.run(run())

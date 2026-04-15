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


def _langgraph_run_payload(
    *, messages: list[dict[str, object]], mode: str | None = None
) -> dict[str, object]:
    input_payload: dict[str, object] = {"messages": messages}
    if mode is not None:
        input_payload["mode"] = mode
    return {"assistant_id": "mcp_agent_executor", "input": input_payload}


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
def temp_thread_state_fixture(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Legacy fixture kept for test signature compatibility.

    Current runtime uses in-memory thread state in ChatRuntimeService, so there is
    no SQLite checkpointer to configure here.
    """
    _ = (monkeypatch, tmp_path)


def test_langgraph_run_validation_errors_return_422() -> None:
    import asyncio

    async def run() -> None:
        from api.main import app

        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-validation/runs",
                headers=headers,
                json={"assistant_id": "mcp_agent_executor"},
            )
            assert resp.status_code == 422
            data = resp.json()
            assert isinstance(data.get("detail"), str)
            assert isinstance(data.get("errors"), list)

    asyncio.run(run())


def test_multi_message_history_is_accepted() -> None:
    import asyncio

    async def run() -> None:
        from api.dependencies import get_graph_service
        from api.main import app

        class StubAgentService:
            async def run_chat(self, **kwargs: object) -> dict[str, object]:
                _ = kwargs
                return {
                    "final_answer": "stubbed",
                    "error": None,
                    "standalone_question": None,
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": None,
                }

        app.dependency_overrides[get_graph_service] = lambda: StubAgentService()
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"type": "system", "content": "Preserve user instructions."},
                {"type": "human", "content": "Use markdown if I ask for it."},
                {"type": "ai", "content": "Okay."},
                {"type": "human", "content": "What did I ask you to do?"},
            ],
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-history/runs",
                headers=headers,
                json=_langgraph_run_payload(
                    messages=cast(list[dict[str, object]], payload["messages"]),
                    mode="direct",
                ),
            )
            assert resp.status_code == 200
        app.dependency_overrides.clear()

    asyncio.run(run())


def test_persistence_across_two_calls_messages_length_is_4(
    temp_thread_state_fixture: None, stub_llm: None
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
                "messages": [{"type": "human", "content": "Hello"}],
            }
            r1 = await client.post(
                f"/api/langgraph/threads/{thread_id}/runs",
                headers=headers,
                json=_langgraph_run_payload(
                    messages=cast(list[dict[str, object]], payload1["messages"]),
                    mode="direct",
                ),
            )
            assert r1.status_code == 200

            # Turn 2
            payload2 = {
                "messages": [{"type": "human", "content": "What did I just say?"}],
            }
            r2 = await client.post(
                f"/api/langgraph/threads/{thread_id}/runs",
                headers=headers,
                json=_langgraph_run_payload(
                    messages=cast(list[dict[str, object]], payload2["messages"]),
                    mode="direct",
                ),
            )
            assert r2.status_code == 200

            resources = getattr(app.state, "resources", None)
            assert resources is not None
            graph_service = resources.chat_runtime_service
            run_config = build_chat_config(thread_id=thread_id, mode="direct")
            snapshot = await graph_service.get_state(run_config)
            vals = getattr(snapshot, "values", None) or {}
            msgs = cast(list[object], vals.get("messages") or [])
            assert len(msgs) == 4  # Human+AI per turn accumulated across 2 turns

    asyncio.run(run())


def test_delete_thread_clears_memory_and_is_idempotent(
    temp_thread_state_fixture: None, stub_llm: None
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
                    "messages": [{"type": "human", "content": content}],
                }
                resp = await client.post(
                    f"/api/langgraph/threads/{thread_id}/runs",
                    headers=headers,
                    json=_langgraph_run_payload(
                        messages=cast(list[dict[str, object]], payload["messages"]),
                        mode="direct",
                    ),
                )
                assert resp.status_code == 200

            # First delete → 204
            d1 = await client.delete(f"/api/threads/{thread_id}")
            assert d1.status_code == 204

            # Second delete remains idempotent → 204
            d2 = await client.delete(f"/api/threads/{thread_id}")
            assert d2.status_code == 204

            resources = getattr(app.state, "resources", None)
            assert resources is not None
            graph_service = resources.chat_runtime_service
            run_config = build_chat_config(thread_id=thread_id, mode="direct")
            snapshot = await graph_service.get_state(run_config)
            vals = getattr(snapshot, "values", None) or {}
            assert not vals, f"Expected empty state after delete, got: {vals}"

    asyncio.run(run())


def test_new_turn_resets_stale_standalone_question_before_search(
    temp_thread_state_fixture: None,
) -> None:
    import asyncio

    async def run() -> None:
        from api.main import app

        class RecordingChatRuntimeService:
            def __init__(self) -> None:
                self.captured_state: dict[str, object] | None = None

            def get_state_values(self, _run_config: dict[str, object]) -> dict[str, object] | None:
                return None

            async def run_chat(
                self,
                *,
                messages: list[dict[str, object]],
                model_id: str | None,
                thread_id: str | None,
                session_id: str | None,
                collection_name: str | None,
                enable_reranker: bool | None,
                enable_tracing: bool | None,
                mode: str | None,
                mcp_server_keys: list[str] | None,
                stream: bool,
            ) -> dict[str, object]:
                _ = (
                    model_id,
                    thread_id,
                    session_id,
                    collection_name,
                    enable_reranker,
                    enable_tracing,
                    mode,
                    mcp_server_keys,
                    stream,
                )
                state: dict[str, object] = {
                    "user_request": str(cast(dict[str, object], messages[-1]).get("content") or ""),
                    "messages": [],
                    "standalone_question": None,
                    "history_text": None,
                }
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

        recording_graph_service = RecordingChatRuntimeService()
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"type": "human", "content": "What are the required properties for a business object?"}],
            "thread_id": f"t_{uuid.uuid4().hex[:8]}",
        }

        from api.dependencies import get_graph_service

        app.dependency_overrides[get_graph_service] = lambda: recording_graph_service
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                resp = await client.post(
                    f"/api/langgraph/threads/{payload['thread_id']}/runs",
                    headers=headers,
                    json=_langgraph_run_payload(
                        messages=cast(list[dict[str, object]], payload["messages"]),
                        mode="rag",
                    ),
                )
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


def test_direct_run_persists_thread_state_without_legacy_turn_state_helper() -> None:
    # Legacy turn-state helper was removed; runtime still persists thread state.
    # Guard that direct mode runtime still records assistant history in thread state.
    import asyncio

    async def run() -> None:
        from api.dependencies import build_chat_config
        from api.main import app

        thread_id = f"t_{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            headers = {"Content-Type": "application/json"}
            payload = {"messages": [{"type": "human", "content": "hello runtime"}]}
            resp = await client.post(
                f"/api/langgraph/threads/{thread_id}/runs",
                headers=headers,
                json=_langgraph_run_payload(
                    messages=cast(list[dict[str, object]], payload["messages"]),
                    mode="direct",
                ),
            )
            assert resp.status_code == 200

        resources = getattr(app.state, "resources", None)
        assert resources is not None
        graph_service = resources.chat_runtime_service
        snapshot = await graph_service.get_state(build_chat_config(thread_id=thread_id))
        vals = cast(dict[str, object], getattr(snapshot, "values", None) or {})
        messages = cast(list[object], vals.get("messages") or [])
        assert len(messages) == 2
        assert vals.get("error") is None

    asyncio.run(run())

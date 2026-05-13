from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.runtime.chat_service import ChatRuntimeService
from src.rag_agent.runtime.thread_checkpoints import LangGraphCheckpointThreadStateStore


def test_chat_runtime_service_restores_thread_state_from_langgraph_checkpoint(tmp_path) -> None:
    db_path = tmp_path / "thread-state.sqlite"
    thread_id = "thread-persistent"

    first_service = ChatRuntimeService(
        thread_state_store=LangGraphCheckpointThreadStateStore(db_path),
    )
    first_service._store_thread_state(
        thread_id,
        [{"role": "user", "content": "Hello"}],
        {
            "final_answer": "Hi",
            "error": None,
            "standalone_question": "Hello",
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
        },
    )

    second_service = ChatRuntimeService(
        thread_state_store=LangGraphCheckpointThreadStateStore(db_path),
    )
    snapshot = asyncio.run(second_service.get_state({"configurable": {"thread_id": thread_id}}))
    messages = snapshot.values["messages"]

    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi"
    assert snapshot.values["final_answer"] == "Hi"


def test_langgraph_checkpoint_store_uses_native_checkpoint_tables(tmp_path) -> None:
    db_path = tmp_path / "thread-state.sqlite"
    thread_id = "thread-checkpoint"
    store = LangGraphCheckpointThreadStateStore(db_path)

    store.put(
        thread_id,
        {
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")],
            "final_answer": "Hi",
        },
    )

    assert store.graph.get_state({"configurable": {"thread_id": thread_id}}).values[
        "final_answer"
    ] == "Hi"
    assert store.get(thread_id)["final_answer"] == "Hi"

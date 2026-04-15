from __future__ import annotations

from src.rag_agent.runtime.agent import RuntimeAgent


def test_runtime_agent_normalize_messages_accepts_langchain_types() -> None:
    messages = RuntimeAgent.normalize_messages(
        [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
            {"type": "system", "content": "rules"},
        ],
        None,
    )

    assert [m.role for m in messages] == ["user", "assistant", "system"]
    assert [m.content for m in messages] == ["hello", "hi", "rules"]


def test_runtime_agent_normalize_messages_falls_back_to_message_field() -> None:
    messages = RuntimeAgent.normalize_messages(None, "fallback")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "fallback"

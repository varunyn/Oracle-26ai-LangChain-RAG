from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    latest_user_message,
)


def test_latest_user_message_reads_native_structured_content() -> None:
    structured = [{"type": "text", "text": "Northway terms are net 30."}]

    result = latest_user_message(
        [
            HumanMessage(content="What are the payment terms?"),
            AIMessage(content=structured),
            HumanMessage(content=[{"type": "text", "text": "What is the deadline?"}]),
        ]
    )

    assert result == "What is the deadline?"


def test_chat_history_before_latest_user_returns_native_messages() -> None:
    history = [
        HumanMessage(content="What are the payment terms?", id="user-1"),
        AIMessage(content=[{"type": "text", "text": "Net 30."}], id="assistant-1"),
        HumanMessage(content="Thanks. What about Summit?", id="user-2"),
    ]

    result = chat_history_before_latest_user(history)

    assert result == history[:2]

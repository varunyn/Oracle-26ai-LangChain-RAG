from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.runtime.memory import (
    langchain_messages_to_dicts,
    latest_user_message,
    merge_chat_messages,
    to_langchain_messages,
)


def test_langchain_messages_to_dicts_preserves_structured_ai_content() -> None:
    structured = [{"type": "text", "text": "Northway terms are net 30."}]

    result = langchain_messages_to_dicts(
        [
            HumanMessage(content="What are the payment terms?"),
            AIMessage(content=structured),
        ]
    )

    assert result == [
        {"role": "user", "content": "What are the payment terms?"},
        {"role": "assistant", "content": structured},
    ]


def test_to_langchain_messages_restores_structured_ai_content() -> None:
    structured = [{"type": "text", "text": "Summit terms are net 45."}]

    result = to_langchain_messages(
        [
            {"role": "user", "content": "Tell me about Summit policies"},
            {"role": "assistant", "content": structured},
        ]
    )

    assert len(result) == 2
    assert result[0].content == "Tell me about Summit policies"
    assert result[1].content == structured


def test_to_langchain_messages_accepts_native_type_aliases() -> None:
    result = to_langchain_messages(
        [
            {"type": "human", "content": "Tell me about Summit policies", "id": "user-1"},
            {"type": "ai", "content": "Summit terms are net 45.", "id": "assistant-1"},
        ]
    )

    assert len(result) == 2
    assert result[0].id == "user-1"
    assert result[0].content == "Tell me about Summit policies"
    assert result[1].id == "assistant-1"
    assert result[1].content == "Summit terms are net 45."


def test_message_id_round_trip_is_preserved() -> None:
    result = to_langchain_messages(
        [
            {"role": "user", "content": "Tell me about Summit policies", "id": "user-1"},
            {"role": "assistant", "content": "Summit terms are net 45.", "id": "assistant-1"},
        ]
    )

    assert result[0].id == "user-1"
    assert result[1].id == "assistant-1"
    assert langchain_messages_to_dicts(result) == [
        {"role": "user", "content": "Tell me about Summit policies", "id": "user-1"},
        {"role": "assistant", "content": "Summit terms are net 45.", "id": "assistant-1"},
    ]


def test_langchain_messages_to_dicts_repairs_legacy_stringified_blocks() -> None:
    legacy = "[{'type': 'text', 'text': 'Northway terms are net 30.'}]"

    result = langchain_messages_to_dicts([AIMessage(content=legacy)])

    assert result == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Northway terms are net 30."}],
        }
    ]


def test_latest_user_message_reads_text_blocks() -> None:
    result = latest_user_message(
        [
            {"role": "user", "content": [{"type": "text", "text": "First question"}]},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": [{"type": "text", "text": "Latest question"}]},
        ]
    )

    assert result == "Latest question"


def test_latest_user_message_accepts_native_human_type() -> None:
    result = latest_user_message(
        [
            {"type": "human", "content": [{"type": "text", "text": "First question"}]},
            {"type": "ai", "content": "First answer"},
            {"type": "human", "content": [{"type": "text", "text": "Latest question"}]},
        ]
    )

    assert result == "Latest question"


def test_merge_chat_messages_replaces_streaming_message_with_same_id() -> None:
    left = [
        HumanMessage(content="Tell me about Summit policies", id="user-1"),
        AIMessage(content="Summit Technologies", id="assistant-1"),
    ]
    right = [
        HumanMessage(content="Tell me about Summit policies", id="user-1"),
        AIMessage(content="Summit Technologies has net 45 terms.", id="assistant-1"),
    ]

    merged = merge_chat_messages(left, right)

    assert len(merged) == 2
    assert merged[0].id == "user-1"
    assert merged[1].id == "assistant-1"
    assert merged[1].content == "Summit Technologies has net 45 terms."


def test_merge_chat_messages_replaces_streaming_dict_message_with_same_id() -> None:
    left = [
        {"role": "user", "content": "Tell me about Summit policies", "id": "user-1"},
        {"role": "assistant", "content": "Summit Technologies", "id": "assistant-1"},
    ]
    right = [
        {"role": "user", "content": "Tell me about Summit policies", "id": "user-1"},
        {
            "role": "assistant",
            "content": "Summit Technologies has net 45 terms.",
            "id": "assistant-1",
        },
    ]

    merged = merge_chat_messages(left, right)

    assert merged == right

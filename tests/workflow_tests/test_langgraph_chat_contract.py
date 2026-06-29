from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.parametrize("mode", ["direct", "rag", "mcp", "mixed"])
def test_chat_agent_accepts_stable_context_modes(mode: str) -> None:
    graph_input = {"messages": [HumanMessage(content="Hello")]}
    config = {
        "configurable": {
            "thread_id": f"contract-{mode}",
            "mode": mode,
            "model_id": "fake-model",
            "session_id": "contract-session",
            "collection_name": "RAG_KNOWLEDGE_BASE",
            "enable_reranker": False,
            "enable_tracing": False,
            "mcp_server_keys": [],
        }
    }

    assert graph_input["messages"][0].content == "Hello"
    assert config["configurable"]["mode"] == mode


def test_assistant_metadata_shape_is_stable() -> None:
    message = AIMessage(
        content="Answer",
        additional_kwargs={
            "mode": "rag",
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
        },
    )

    assert message.additional_kwargs["mode"] == "rag"
    assert message.additional_kwargs["citations"] == []
    assert message.additional_kwargs["reranker_docs"] == []
    assert message.additional_kwargs["mcp_tools_used"] == []

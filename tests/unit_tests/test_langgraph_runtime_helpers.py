from __future__ import annotations

from langchain_core.messages import AIMessage

from src.rag_agent.graphs import runtime as graph_runtime
from src.rag_agent.graphs.runtime import result_to_assistant_message


def test_build_run_config_places_chat_context_under_configurable() -> None:
    config = graph_runtime.build_run_config(
        parent_config={"callbacks": ["graph-handler"]},
        thread_id="thread-1",
        mode="rag",
        model_id="model-1",
        session_id="session-1",
        enable_tracing=True,
        mcp_server_keys=["oracle"],
    )

    configurable = config["configurable"]
    assert configurable["thread_id"] == "thread-1"
    assert configurable["mode"] == "rag"
    assert configurable["model_id"] == "model-1"
    assert configurable["session_id"] == "session-1"
    assert configurable["enable_tracing"] is True
    assert configurable["mcp_server_keys"] == ["oracle"]
    assert config["callbacks"] == ["graph-handler"]
    assert config["metadata"]["session_id"] == "session-1"


def test_build_run_config_propagates_request_identity_and_release() -> None:
    config = graph_runtime.build_run_config(
        thread_id="thread-1",
        mode="mixed",
        model_id="model-1",
        session_id="session-1",
        request_id="request-1",
        user_id="user-1",
        release="release-1",
        enable_tracing=True,
        mcp_server_keys=None,
    )

    configurable = config["configurable"]
    metadata = config["metadata"]
    assert configurable["request_id"] == "request-1"
    assert configurable["user_id"] == "user-1"
    assert configurable["release"] == "release-1"
    assert metadata["request_id"] == "request-1"
    assert metadata["thread_id"] == "thread-1"
    assert metadata["session_id"] == "session-1"
    assert metadata["user_id"] == "user-1"
    assert metadata["release"] == "release-1"


def test_result_to_assistant_message_preserves_reference_payload() -> None:
    message = result_to_assistant_message(
        "mixed",
        {
            "final_answer": "Answer",
            "standalone_question": "Question",
            "citations": [{"source": "doc.md"}],
            "reranker_docs": [],
            "context_usage": {"chunks": 1},
            "mcp_used": True,
            "mcp_tools_used": ["lookup"],
            "mcp_tool_invocations": [{"tool_name": "lookup"}],
            "trace_id": "trace-1",
            "error": None,
        },
    )

    assert isinstance(message, AIMessage)
    assert message.content == "Answer"
    assert message.additional_kwargs["mode"] == "mixed"
    assert message.additional_kwargs["standalone_question"] == "Question"
    assert message.additional_kwargs["citations"] == [{"source": "doc.md"}]
    assert message.additional_kwargs["mcp_used"] is True
    assert message.additional_kwargs["mcp_tools_used"] == ["lookup"]
    assert message.additional_kwargs["trace_id"] == "trace-1"

import json
from importlib import import_module
from pathlib import Path
from typing import Any


def test_langgraph_json_registers_chat_agent() -> None:
    config = json.loads(Path("langgraph.json").read_text())
    assert config["dependencies"] == ["."]
    assert config["graphs"]["chat_agent"] == "./src/rag_agent/graphs/chat_agent.py:make_chat_agent"
    assert config["http"]["app"] == "./api/main.py:app"
    assert config["env"] == ".env"

    graph_module = import_module("src.rag_agent.graphs.chat_agent")
    api_module = import_module("api.main")

    assert graph_module.chat_agent is not None
    assert callable(graph_module.build_chat_agent)
    assert callable(graph_module.make_chat_agent)
    assert api_module.app is not None


def test_make_chat_agent_attaches_one_callback_configuration(monkeypatch: Any) -> None:
    graph_module = import_module("src.rag_agent.graphs.chat_agent")
    captured: dict[str, object] = {}

    def fake_add_callbacks(run_config: dict[str, object], **kwargs: object) -> None:
        captured["run_config"] = run_config
        captured["kwargs"] = kwargs
        run_config["callbacks"] = ["handler"]

    monkeypatch.setattr(graph_module, "add_langfuse_callbacks", fake_add_callbacks)

    graph = graph_module.make_chat_agent(
        {
            "configurable": {
                "enable_tracing": True,
                "mode": "mixed",
                "model_id": "model-a",
                "thread_id": "thread-1",
                "session_id": "session-1",
            }
        }
    )

    assert graph is not None
    assert graph.config["callbacks"] == ["handler"]
    assert captured["run_config"] == {
        "configurable": {
            "enable_tracing": True,
            "mode": "mixed",
            "model_id": "model-a",
            "thread_id": "thread-1",
            "session_id": "session-1",
        },
        "callbacks": ["handler"],
    }
    assert captured["kwargs"] == {
        "session_id": "session-1",
        "request_id": None,
        "user_id": None,
        "release": None,
        "trace_name": "chat.request",
        "tags": ["chat", "mode:mixed", "model:model-a"],
    }

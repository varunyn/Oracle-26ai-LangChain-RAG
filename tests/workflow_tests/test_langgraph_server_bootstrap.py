import json
from importlib import import_module
from pathlib import Path


def test_langgraph_json_registers_chat_agent() -> None:
    config = json.loads(Path("langgraph.json").read_text())
    assert config["dependencies"] == ["."]
    assert config["graphs"]["chat_agent"] == "./src/rag_agent/graphs/chat_agent.py:chat_agent"
    assert config["http"]["app"] == "./api/main.py:app"
    assert config["env"] == ".env"

    graph_module = import_module("src.rag_agent.graphs.chat_agent")
    api_module = import_module("api.main")

    assert graph_module.chat_agent is not None
    assert callable(graph_module.build_chat_agent)
    assert api_module.app is not None

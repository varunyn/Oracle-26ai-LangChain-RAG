import json
from pathlib import Path


def test_langgraph_json_registers_chat_agent() -> None:
    config = json.loads(Path("langgraph.json").read_text())
    assert config["graphs"]["chat_agent"] == "./src/rag_agent/graphs/chat_agent.py:chat_agent"
    assert config["env"] == ".env"

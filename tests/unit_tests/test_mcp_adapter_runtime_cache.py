import asyncio
from types import SimpleNamespace

from src.rag_agent.infrastructure import mcp_adapter_runtime as mod


def test_clear_adapter_runtime_cache_clears_cached_clients_and_tools() -> None:
    class FakeTool:
        def __init__(self) -> None:
            self.name = "calculator.fake"
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake_tool = FakeTool()
    mod._client_cache["cfg"] = SimpleNamespace()
    mod._tool_cache["cfg"] = [fake_tool]

    asyncio.run(mod.clear_adapter_runtime_cache())

    assert fake_tool.closed is True
    assert mod._client_cache == {}
    assert mod._tool_cache == {}

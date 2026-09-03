import asyncio
from types import SimpleNamespace

from src.rag_agent.infrastructure import mcp_adapter_runtime as mod


def test_clear_adapter_runtime_cache_clears_cached_clients_and_tools() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_client = FakeClient()
    mod._client_cache["cfg"] = SimpleNamespace(client=fake_client)
    mod._tool_cache["cfg"] = [SimpleNamespace(name="calculator.fake")]

    asyncio.run(mod.clear_adapter_runtime_cache())

    assert fake_client.closed is True
    assert mod._client_cache == {}
    assert mod._tool_cache == {}

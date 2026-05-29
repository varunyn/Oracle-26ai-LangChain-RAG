from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from .mcp_adapter_runtime import load_adapter_tools


async def get_mcp_tools_async(
    *,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> list[BaseTool]:
    return await load_adapter_tools(server_keys=server_keys, run_config=run_config)


__all__ = [
    "get_mcp_tools_async",
    "load_adapter_tools",
]

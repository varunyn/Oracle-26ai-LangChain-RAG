from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import cast

from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field
from typing_extensions import override

from .code_mode_client import get_code_mode_client

logger = logging.getLogger(__name__)


class CallToolChainInput(BaseModel):
    code: str = Field(..., description="Code to execute in code-mode tool chain.")
    timeout: int | None = Field(
        default=None,
        description="Timeout in seconds for the tool chain execution.",
    )


def _run_coroutine_in_thread(
    coro: Coroutine[object, object, dict[str, object]],
) -> dict[str, object]:
    result: dict[str, dict[str, object]] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("Thread runner did not return a value.")
    return result["value"]


def _log_code_mode_invocation(code: str) -> None:
    """Log whether executed code mentions OCI MCP (for debugging: did the model call run_oci_command?)."""
    code_lower = code.lower()
    mentions_oci = (
        "run_oci_command" in code_lower
        or "get_oci_command_help" in code_lower
        or "oci-mcp-server" in code_lower
    )
    logger.info(
        "call_tool_chain code_len=%d mentions_oci_mcp=%s snippet=%s",
        len(code),
        mentions_oci,
        code[:300].replace("\n", " ") if code else "",
    )


def _run_call_tool_chain_sync(code: str, timeout: int | None) -> dict[str, object]:
    _log_code_mode_invocation(code)
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        result = asyncio.run(get_code_mode_client().call_tool_chain(code, timeout=timeout or 30))
        return cast(dict[str, object], result)
    return _run_coroutine_in_thread(
        get_code_mode_client().call_tool_chain(code, timeout=timeout or 30)
    )


class CallToolChainTool(BaseTool):
    name: str = "call_tool_chain"
    description: str = (
        "Execute code in the code-mode tool chain (has access to MCP tools; use when the user "
        "asks for data that could come from commands, APIs, or external tools, e.g. account/tenancy info)."
    )
    args_schema: ArgsSchema | None = CallToolChainInput

    @override
    def _run(self, code: str, timeout: int | None = None) -> dict[str, object]:
        return _run_call_tool_chain_sync(code=code, timeout=timeout)

    @override
    async def _arun(self, code: str, timeout: int | None = None) -> dict[str, object]:
        _log_code_mode_invocation(code)
        result = await get_code_mode_client().call_tool_chain(code, timeout=timeout or 30)
        return cast(dict[str, object], result)


call_tool_chain_tool = CallToolChainTool(
    description=(
        "Execute code in the code-mode tool chain (has access to MCP tools; use when the user "
        "asks for data that could come from commands, APIs, or external tools, e.g. account/tenancy info)."
    )
)

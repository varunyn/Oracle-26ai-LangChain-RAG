import asyncio
import concurrent.futures
import logging
from collections.abc import Coroutine
from typing import TypeVar, cast

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import override

from api.settings import get_settings

from ...agent_state import State, messages_text_from_state
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.tool_selection import select_mcp_tools_for_question_async

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _run_async(coro: Coroutine[object, object, T]) -> T:
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class Router(Runnable[State, dict[str, object | None]]):
    """Routes to search (RAG), select_mcp, or direct based on configurable.mode."""

    @override
    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object | None]:
        return _run_async(self._route_async(input, config))

    @override
    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object | None]:
        return await self._route_async(input, config)

    async def _route_async(
        self, input: State, config: RunnableConfig | None = None
    ) -> dict[str, object | None]:
        log_node_start("Router")
        if config is None:
            run_config: RunnableConfig = cast(RunnableConfig, cast(object, {}))
        else:
            run_config = config
        configurable: dict[str, object] = run_config.get("configurable") or {}
        mode = str(configurable.get("mode") or "rag").lower().strip()

        state_update: dict[str, object | None] = {
            "route": None,
            "mode": mode,
            "history_text": messages_text_from_state(input),
        }

        if mode == "direct":
            route = "direct"
            logger.info("Router: mode=%s → route=%s (direct answer, no RAG/MCP)", mode, route)
            state_update["route"] = route
            log_node_end("Router", route=route)
            return state_update
        elif mode == "mcp":
            route = "select_mcp"
            logger.info("Router: mode=%s → route=%s (MCP only, no RAG)", mode, route)
            state_update.update(
                {
                    "route": route,
                    "rag_answer": None,
                    "retriever_docs": [],
                    "reranker_docs": [],
                    "citations": [],
                }
            )
            log_node_end("Router", route=route)
            return state_update
        elif mode == "mixed":
            question = str(input.get("user_request") or "").strip()
            _settings = get_settings()
            max_tools = _settings.MCP_TOOL_SELECTION_MAX_TOOLS or 5
            server_keys_val = configurable.get("mcp_server_keys")
            server_keys = (
                [str(item).strip() for item in server_keys_val if str(item).strip()]
                if isinstance(server_keys_val, list)
                else None
            )
            selected_tool_names: list[str] = []
            selected_tool_descriptions: list[str] = []
            try:
                selection = await select_mcp_tools_for_question_async(
                    question,
                    limit=max_tools,
                    server_keys=server_keys,
                    run_config=run_config,
                )
                for item in selection.get("selected_tools") or []:
                    if not isinstance(item, dict):
                        continue
                    canonical_name = item.get("canonical_name")
                    description = item.get("description")
                    if isinstance(canonical_name, str) and canonical_name.strip():
                        selected_tool_names.append(canonical_name.strip())
                    if isinstance(description, str) and description.strip():
                        selected_tool_descriptions.append(description.strip())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Router: mixed tool selection failed (%s); routing to RAG", exc)

            route_val = "search"
            state_update["route"] = route_val
            state_update["mode"] = "mixed"
            state_update["mcp_tool_match"] = bool(selected_tool_names)
            state_update["selected_mcp_tool_names"] = selected_tool_names
            state_update["selected_mcp_tool_descriptions"] = selected_tool_descriptions
            logger.info(
                "Router: mode=mixed → route=%s (RAG first, tool_match=%s, selected_tools=%s)",
                route_val,
                bool(selected_tool_names),
                selected_tool_names,
            )
            log_node_end("Router", route=route_val)
            return state_update
        else:
            route = "search"
            logger.info("Router: mode=%s → route=%s (RAG path)", mode, route)
            state_update["route"] = route
            log_node_end("Router", route=route)
            return state_update

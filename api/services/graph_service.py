"""GraphService: boundary between FastAPI and LangGraph.

Owns building/holding the compiled agent graph and provides simple
invoke/stream/state access methods for routers via DI.

Keep creation side effects minimal (no DB connections at import).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from src.rag_agent import create_workflow


class GraphService:
    """Small service to wrap the LangGraph compiled workflow.

    Singleton-ish per process: construct once and reuse via FastAPI Depends provider.
    When graph is provided (e.g. from lifespan with AsyncSqliteSaver), uses it;
    otherwise builds workflow with default sync checkpointer (e.g. tests).
    """

    def __init__(self, graph: Any = None) -> None:
        self._graph: Any = graph if graph is not None else create_workflow()

    # --- Basic accessors ---
    def get_graph(self) -> Any:
        """Return the underlying graph object (for visualization only)."""
        return self._graph.get_graph()

    # --- Invocation APIs ---
    def invoke(self, state: Any, run_config: dict[str, Any]) -> dict[str, Any]:
        """Run a non-stream invocation and return the final state values dict."""
        # langgraph returns a dict-like state for invoke()
        return self._graph.invoke(state, config=run_config)  # type: ignore

    async def astream(
        self, state: Any, run_config: dict[str, Any], *, stream_mode: list[str] | None = None
    ) -> AsyncIterator[Any]:
        """Stream workflow events with fallback for sync-only checkpointers.

        Preferred path uses graph.astream(). If the underlying checkpointer does not
        implement async methods (e.g., SqliteSaver), fall back to driving the sync
        graph.stream() in a background thread and yielding events asynchronously.
        """
        modes = stream_mode or ["updates", "messages"]

        # Try native async streaming first
        try:
            async for event in self._graph.astream(  # type: ignore
                state, config=run_config, stream_mode=modes
            ):
                yield event
            return
        except Exception as exc:  # Fallback for sync-only checkpointers
            msg = str(exc)
            fallback_needed = (
                isinstance(exc, NotImplementedError)
                or ("does not support async methods" in msg)
                or ("SqliteSaver" in msg and "async" in msg)
                or ("aget_" in msg)
            )
            if not fallback_needed:
                raise

        # Fallback: run sync stream() in a dedicated thread and forward events via an asyncio.Queue
        async_queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            err: BaseException | None = None
            try:
                for ev in self._graph.stream(  # type: ignore
                    state, config=run_config, stream_mode=modes
                ):
                    # Hand off event safely into the asyncio loop
                    _ = loop.call_soon_threadsafe(async_queue.put_nowait, ev)
            except BaseException as e:  # Capture any producer error to raise on consumer side
                err = e
            finally:
                # Signal completion with optional error
                _ = loop.call_soon_threadsafe(async_queue.put_nowait, (sentinel, err))

        t = threading.Thread(target=_producer, name="GraphStreamSyncProducer", daemon=True)
        t.start()

        while True:
            item = await async_queue.get()
            if isinstance(item, tuple) and len(item) == 2 and item[0] is sentinel:
                _sentinel, err = item
                if err:
                    raise err
                break
            yield item

    async def get_state(self, run_config: dict[str, Any]) -> Any:
        """Expose graph state for reading final values after a stream run.

        Uses aget_state when the checkpointer is async (e.g. AsyncSqliteSaver)
        so we avoid InvalidStateError when called from the main async thread.
        Falls back to sync get_state when the checkpointer does not support async
        (e.g. SqliteSaver when lifespan did not run, e.g. in tests with ASGITransport).
        """
        if hasattr(self._graph, "aget_state"):
            try:
                return await self._graph.aget_state(run_config)  # type: ignore
            except (NotImplementedError, Exception) as e:
                msg = str(e)
                if "does not support async" in msg or "SqliteSaver" in msg and "async" in msg:
                    return self._graph.get_state(run_config)  # type: ignore
                raise
        return self._graph.get_state(run_config)  # type: ignore

    async def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for the given thread_id.

        Uses adelete_thread when the checkpointer is async; otherwise sync delete_thread
        (e.g. SqliteSaver when lifespan did not run, e.g. in tests with ASGITransport).
        """
        checkpointer = self._graph.checkpointer
        if hasattr(checkpointer, "adelete_thread"):
            try:
                await checkpointer.adelete_thread(thread_id)
            except NotImplementedError:
                checkpointer.delete_thread(thread_id)
            except Exception as e:
                msg = str(e)
                if "does not support async" in msg or ("SqliteSaver" in msg and "async" in msg):
                    checkpointer.delete_thread(thread_id)
                else:
                    raise
        else:
            checkpointer.delete_thread(thread_id)

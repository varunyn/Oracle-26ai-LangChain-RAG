from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .thread_checkpoint_state import ThreadCheckpointState


def _persist_state(state: ThreadCheckpointState) -> ThreadCheckpointState:
    return state


class LangGraphCheckpointThreadStateStore:
    """Persist current runtime thread state through LangGraph checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._checkpointer.setup()
        builder = StateGraph(ThreadCheckpointState)
        builder.add_node("persist_thread_state", _persist_state)
        builder.add_edge(START, "persist_thread_state")
        builder.add_edge("persist_thread_state", END)
        self.graph = builder.compile(checkpointer=self._checkpointer)

    def get(self, thread_id: str) -> dict[str, Any] | None:
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(cast(dict[str, Any], snapshot.values or {}))
        return values or None

    def put(self, thread_id: str, state: dict[str, Any]) -> None:
        cast(Any, self.graph).invoke(cast(ThreadCheckpointState, state), self._config(thread_id))

    def delete(self, thread_id: str) -> None:
        self._checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})


__all__ = ["LangGraphCheckpointThreadStateStore"]

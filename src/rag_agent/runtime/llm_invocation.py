from __future__ import annotations

from collections.abc import Sequence

from langchain_core.runnables.config import RunnableConfig


def invoke_llm_with_optional_config(
    llm: object,
    messages: Sequence[object],
    run_config: RunnableConfig | None,
) -> object:
    invoke = getattr(llm, "invoke")
    if run_config:
        try:
            return invoke(messages, config=run_config)
        except TypeError:
            return invoke(messages)
    return invoke(messages)


__all__ = ["invoke_llm_with_optional_config"]

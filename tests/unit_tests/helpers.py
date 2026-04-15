from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def fake_chat_model(messages: Iterable[AIMessage | str]) -> GenericFakeChatModel:
    """Build a deterministic LangChain fake chat model for unit tests."""

    return GenericFakeChatModel(messages=iter(messages))


def tool_call_message(name: str, args: dict[str, object], tool_call_id: str) -> AIMessage:
    """Build a deterministic AIMessage containing a single tool call."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": tool_call_id,
            }
        ],
    )


class ToolBindableFakeChatModel:
    """Adapter for tests where production code expects bind_tools()."""

    def __init__(self, messages: Iterable[AIMessage | str]) -> None:
        self._model = fake_chat_model(messages)
        self.bind_calls: list[dict[str, object]] = []
        self.seen_messages: list[list[object]] = []

    def bind_tools(self, tools: list[object], **kwargs: object) -> ToolBindableFakeChatModel:
        self.bind_calls.append({"tools": tools, "kwargs": kwargs})
        return self

    def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
        _ = config
        self.seen_messages.append(messages)
        return AIMessage.model_validate(self._model.invoke(messages))


class StructuredOutputFakeChatModel:
    """Adapter for tests where production code expects with_structured_output()."""

    def __init__(self, structured_responses: Iterable[object], raw_messages: Iterable[AIMessage | str] = ()) -> None:
        self._structured_responses = iter(structured_responses)
        self._raw_model = fake_chat_model(raw_messages)
        self.schemas: list[type | dict[str, Any]] = []
        self.include_raw_values: list[bool] = []
        self.seen_messages: list[list[object]] = []

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> StructuredOutputFakeChatModel:
        _ = kwargs
        self.schemas.append(schema)
        self.include_raw_values.append(include_raw)
        return self

    def invoke(self, messages: list[object], *, config: object | None = None) -> object:
        _ = config
        self.seen_messages.append(messages)
        if self.schemas:
            return next(self._structured_responses)
        return AIMessage.model_validate(self._raw_model.invoke(messages))

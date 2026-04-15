from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class AnswerSchema(BaseModel):
    answer: str


def test_generic_fake_chat_model_supports_with_structured_output_for_pydantic() -> None:
    model = GenericFakeChatModel(messages=iter(['{"answer": "structured ok"}']))

    with pytest.raises(NotImplementedError, match="with_structured_output is not implemented"):
        structured = model.with_structured_output(AnswerSchema)
        structured.invoke([HumanMessage(content="hello")])

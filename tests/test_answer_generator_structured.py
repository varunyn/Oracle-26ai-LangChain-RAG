from typing import cast
from unittest.mock import MagicMock, patch

from src.rag_agent.agent_state import DocSerializable, State
from src.rag_agent.langgraph.nodes.answer_generator import (
    STRUCTURED_FAILED_MESSAGE,
    AnswerFromDocs,
)
from src.rag_agent.prompts import ANSWER_STRUCTURED_PROMPT_TEMPLATE
from src.rag_agent.schemas import (
    StructuredRAGAnswer,
    extract_inline_citation_ids,
    validate_inline_citation_ids,
    validate_structured_markdown_answer,
)


def test_extract_inline_citation_ids_preserves_first_seen_order():
    markdown = "Alpha [2] beta [1] again [2] then [3]."
    assert extract_inline_citation_ids(markdown) == [2, 1, 3]


def test_validate_inline_citation_ids_filters_invalid_and_duplicates():
    citation_ids = [2, 99, 2, 1, 0, 3]
    assert validate_inline_citation_ids(citation_ids, 3) == [2, 1, 3]


def test_validate_structured_markdown_answer_intersects_declared_with_inline():
    answer = StructuredRAGAnswer(
        markdown="- First [2]\n- Second [1]\n- Third [9]",
        valid_citation_ids=[1, 2, 3],
    )
    markdown, citation_ids = validate_structured_markdown_answer(answer, 3)

    assert markdown == "- First [2]\n- Second [1]\n- Third [9]"
    assert citation_ids == [2, 1]


def test_validate_structured_markdown_answer_uses_inline_ids_when_declared_empty():
    answer = StructuredRAGAnswer(
        markdown="Paragraph with [1] and [2].",
        valid_citation_ids=[],
    )
    markdown, citation_ids = validate_structured_markdown_answer(answer, 2)

    assert markdown == "Paragraph with [1] and [2]."
    assert citation_ids == [1, 2]


def test_answer_from_docs_zero_docs_skips_llm():
    node = AnswerFromDocs()
    state: State = {
        "user_request": "What is X?",
        "messages": [],
        "reranker_docs": [],
        "citations": [],
    }
    with patch("src.rag_agent.langgraph.nodes.answer_generator.get_llm") as mock_get_llm:
        result = node.invoke(state, config={})
    mock_get_llm.assert_not_called()
    assert "don't know" in cast(str, result["rag_answer"]).lower()
    assert result.get("citations") == []


def test_answer_from_docs_structured_path_preserves_markdown_formatting():
    node = AnswerFromDocs()
    docs: list[DocSerializable] = [
        {"page_content": "Chunk one."},
        {"page_content": "Chunk two."},
    ]
    state: State = {
        "user_request": "Answer as a numbered list",
        "messages": [],
        "reranker_docs": docs,
        "citations": [{"source": "a"}, {"source": "b"}],
    }
    structured_result = StructuredRAGAnswer(
        markdown="1. First item [1]\n2. Second item [2]",
        valid_citation_ids=[1, 2],
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = structured_result
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.rag_agent.langgraph.nodes.answer_generator.get_llm", return_value=mock_llm):
        with patch(
            "src.rag_agent.langgraph.nodes.answer_generator.calculate_context_usage",
            return_value={"input_tokens": 0, "output_tokens": 0},
        ):
            result = node.invoke(state, config={"configurable": {}})

    rag_answer = cast(str, result["rag_answer"])
    assert rag_answer == "1. First item [1]\n2. Second item [2]"
    mock_llm.with_structured_output.assert_called_once()
    mock_structured.invoke.assert_called_once()


def test_answer_structured_prompt_prioritizes_latest_user_instruction_generically():
    assert "latest user message is the controlling instruction" in ANSWER_STRUCTURED_PROMPT_TEMPLATE
    assert (
        "keep the same topic and facts but present them according to that latest instruction"
        in (ANSWER_STRUCTURED_PROMPT_TEMPLATE)
    )


def test_answer_from_docs_structured_path_rejects_missing_valid_citations():
    node = AnswerFromDocs()
    docs: list[DocSerializable] = [{"page_content": "Only chunk."}]
    state: State = {
        "user_request": "What is it?",
        "messages": [],
        "reranker_docs": docs,
        "citations": [{"source": "a"}],
    }
    structured_result = StructuredRAGAnswer(
        markdown="Answer with invalid citation [9]",
        valid_citation_ids=[9],
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = structured_result
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.rag_agent.langgraph.nodes.answer_generator.get_llm", return_value=mock_llm):
        with patch(
            "src.rag_agent.langgraph.nodes.answer_generator.calculate_context_usage",
            return_value={"input_tokens": 0, "output_tokens": 0},
        ):
            result = node.invoke(state, config={"configurable": {}})

    assert cast(str, result["rag_answer"]) == STRUCTURED_FAILED_MESSAGE


def test_answer_from_docs_structured_fail_returns_message():
    node = AnswerFromDocs()
    docs: list[DocSerializable] = [{"page_content": "Only chunk."}]
    state: State = {
        "user_request": "What is it?",
        "messages": [],
        "reranker_docs": docs,
        "citations": [{"source": "a"}],
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="This is just plain text.")
    mock_llm.stream.return_value = [MagicMock(content="Never used.")]

    with patch("src.rag_agent.langgraph.nodes.answer_generator.get_llm", return_value=mock_llm):
        with patch(
            "src.rag_agent.langgraph.nodes.answer_generator.calculate_context_usage",
            return_value={"input_tokens": 0, "output_tokens": 0},
        ):
            result = node.invoke(state, config={"configurable": {}})

    assert cast(str, result["rag_answer"]) == STRUCTURED_FAILED_MESSAGE
    mock_llm.stream.assert_not_called()

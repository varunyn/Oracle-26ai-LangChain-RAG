from typing import cast
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.agent_state import State


def test_router_history_text_is_available_for_followup_interpretation() -> None:
    from src.rag_agent.langgraph.nodes.router import Router

    router = Router()
    state: State = {
        "user_request": "give me that answer in bullet points",
        "messages": [
            HumanMessage(content="How do I create a new visual application?"),
            AIMessage(content="Create it from the Visual Applications page by clicking New. [1]"),
            HumanMessage(content="give me that answer in bullet points"),
        ],
    }

    result = router.invoke(state, config={"configurable": {"mode": "rag"}})

    history_text = cast(str, result.get("history_text") or "")
    assert "How do I create a new visual application?" in history_text
    assert "give me that answer in bullet points" in history_text


def test_followup_interpreter_marks_formatting_request_without_hardcoded_branching() -> None:
    from src.rag_agent.langgraph.nodes.followup_interpreter import FollowUpInterpreter

    interpreter = FollowUpInterpreter()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"intent":"reformat","standalone_question":null,"response_instruction":"Rewrite the last grounded answer as bullet points.","reasoning":"The user asked only for presentation changes."}'
    )

    state: State = {
        "user_request": "give me that answer in bullet points",
        "messages": [
            HumanMessage(content="How do I create a new visual application?"),
            AIMessage(content="Create it from the Visual Applications page by clicking New. [1]"),
            HumanMessage(content="give me that answer in bullet points"),
        ],
        "history_text": "human: How do I create a new visual application?\nai: Create it from the Visual Applications page by clicking New. [1]\nhuman: give me that answer in bullet points",
        "rag_answer": "Create it from the Visual Applications page by clicking New. [1]",
        "citations": [{"source": "doc-1"}],
    }

    with patch("src.rag_agent.langgraph.nodes.followup_interpreter.get_llm", return_value=mock_llm):
        result = interpreter.invoke(
            state, config={"configurable": {"model_id": "meta.llama-4-scout-17b-16e-instruct"}}
        )

    assert result["followup_intent"] == "reformat"
    assert result["response_instruction"] == "Rewrite the last grounded answer as bullet points."
    assert result["standalone_question"] is None


def test_followup_interpreter_rewrites_contextual_followup_for_retrieval() -> None:
    from src.rag_agent.langgraph.nodes.followup_interpreter import FollowUpInterpreter

    interpreter = FollowUpInterpreter()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"intent":"retrieve","standalone_question":"What happens if I enter a duplicate Application Name when creating a new visual application in Oracle Visual Builder?","response_instruction":null,"reasoning":"The follow-up asks for new information tied to the prior topic."}'
    )

    state: State = {
        "user_request": "What happens if I enter a duplicate Application Name?",
        "messages": [
            HumanMessage(content="How do I create a new visual application?"),
            AIMessage(content="Create it from the Visual Applications page by clicking New. [1]"),
            HumanMessage(content="What happens if I enter a duplicate Application Name?"),
        ],
        "history_text": "human: How do I create a new visual application?\nai: Create it from the Visual Applications page by clicking New. [1]\nhuman: What happens if I enter a duplicate Application Name?",
    }

    with patch("src.rag_agent.langgraph.nodes.followup_interpreter.get_llm", return_value=mock_llm):
        result = interpreter.invoke(
            state, config={"configurable": {"model_id": "meta.llama-4-scout-17b-16e-instruct"}}
        )

    assert result["followup_intent"] == "retrieve"
    assert result["standalone_question"] == (
        "What happens if I enter a duplicate Application Name when creating a new visual application in Oracle Visual Builder?"
    )
    assert result["response_instruction"] is None


def test_followup_interpreter_uses_mcp_answer_as_latest_grounded_answer() -> None:
    from src.rag_agent.langgraph.nodes.followup_interpreter import FollowUpInterpreter

    interpreter = FollowUpInterpreter()
    captured_prompt: dict[str, str] = {}

    mock_llm = MagicMock()

    def fake_invoke(messages, config=None):
        _ = config
        captured_prompt["content"] = messages[0].content
        return MagicMock(
            content='{"intent":"reformat","standalone_question":null,"response_instruction":"Restate the result plainly.","reasoning":"The user is asking for the previous tool result in plain language."}'
        )

    mock_llm.invoke.side_effect = fake_invoke

    state: State = {
        "user_request": "ok and what's the answer?",
        "messages": [
            HumanMessage(content="Solve the equation x^2 - 5x + 6 = 0 use tools"),
            AIMessage(content='calculator.solve_equation(equation="x**2 - 5*x + 6 = 0")'),
            HumanMessage(content="ok and what's the answer?"),
        ],
        "history_text": "human: Solve the equation x^2 - 5x + 6 = 0 use tools\nai: calculator.solve_equation(equation=\"x**2 - 5*x + 6 = 0\")\nhuman: ok and what's the answer?",
        "mcp_answer": "The solutions are x = 2 and x = 3.",
    }

    with patch("src.rag_agent.langgraph.nodes.followup_interpreter.get_llm", return_value=mock_llm):
        result = interpreter.invoke(
            state, config={"configurable": {"model_id": "meta.llama-4-scout-17b-16e-instruct"}}
        )

    assert "Latest grounded answer: The solutions are x = 2 and x = 3." in captured_prompt["content"]
    assert result["followup_intent"] == "reformat"


def test_followup_interpreter_parse_failure_routes_mcp_followup_when_mcp_answer_exists() -> None:
    from src.rag_agent.langgraph.nodes.followup_interpreter import FollowUpInterpreter

    interpreter = FollowUpInterpreter()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not json")

    state: State = {
        "user_request": "ok and what's the answer?",
        "messages": [],
        "history_text": "human: Solve the equation\nai: calculator.solve_equation(...)\nhuman: ok and what's the answer?",
        "mcp_answer": "The solutions are x = 2 and x = 3.",
        "mode": "mixed",
        "mcp_tool_match": True,
    }

    with patch("src.rag_agent.langgraph.nodes.followup_interpreter.get_llm", return_value=mock_llm):
        result = interpreter.invoke(
            state, config={"configurable": {"model_id": "meta.llama-4-scout-17b-16e-instruct"}}
        )

    assert result["followup_intent"] == "mcp_followup"
    assert result["standalone_question"] is None
    assert result["response_instruction"] is None


def test_grounded_reformat_answer_uses_existing_grounded_answer_without_retrieval() -> None:
    from src.rag_agent.langgraph.nodes.followup_interpreter import GroundedReformatAnswer

    node = GroundedReformatAnswer()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="- Open the Visual Applications page. [1]\n- Click **New**. [1]"
    )

    state: State = {
        "user_request": "give me that answer in bullet points",
        "messages": [
            HumanMessage(content="How do I create a new visual application?"),
            AIMessage(content="Create it from the Visual Applications page by clicking New. [1]"),
            HumanMessage(content="give me that answer in bullet points"),
        ],
        "history_text": "human: How do I create a new visual application?\nai: Create it from the Visual Applications page by clicking New. [1]\nhuman: give me that answer in bullet points",
        "rag_answer": "Create it from the Visual Applications page by clicking New. [1]",
        "citations": [{"source": "doc-1"}],
        "response_instruction": "Rewrite the last grounded answer as bullet points.",
    }

    with patch("src.rag_agent.langgraph.nodes.followup_interpreter.get_llm", return_value=mock_llm):
        result = node.invoke(
            state, config={"configurable": {"model_id": "meta.llama-4-scout-17b-16e-instruct"}}
        )

    assert result["rag_answer"] == "- Open the Visual Applications page. [1]\n- Click **New**. [1]"
    assert result["rag_has_citations"] is True

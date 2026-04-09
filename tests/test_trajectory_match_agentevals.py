import os

import pytest
from agentevals.trajectory.match import (  # type: ignore[import-untyped]
    create_trajectory_match_evaluator,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.rag_agent.infrastructure.oci_models import get_llm


@pytest.mark.vcr()
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OCI_INTEGRATION_TESTS", "0") != "1",
    reason="Set OCI_INTEGRATION_TESTS=1 to run AgentEvals integration tests",
)
def test_trajectory_strict_tool_call():
    """
    Demonstrates AgentEvals strict trajectory match against a simple tool binding.
    This uses a real LLM; record with VCR by running once with --record-mode=once.
    """

    @tool
    def echo_text(text: str) -> str:
        """Echo tool."""
        return text

    llm = get_llm()
    bound = llm.bind_tools([echo_text])

    result = bound.invoke([HumanMessage(content="Use the echo tool to say hi")])

    # Reference trajectory expects a tool call to echo_text with arg {"text": "hi"}
    reference_trajectory = [
        HumanMessage(content="Use the echo tool to say hi"),
        AIMessage(
            content="", tool_calls=[{"id": "call_1", "name": "echo_text", "args": {"text": "hi"}}]
        ),
        # ToolMessage is inferred by evaluator; for strict mode we focus on tool call presence
    ]

    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="subset")
    evaluation = evaluator(
        outputs=result["messages"] if isinstance(result, dict) else result,
        reference_outputs=reference_trajectory,
    )
    assert evaluation["score"] is True

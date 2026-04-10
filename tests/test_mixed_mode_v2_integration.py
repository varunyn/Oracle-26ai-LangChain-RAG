import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from src.rag_agent.langgraph.graph import create_workflow


def test_v2_workflow_retrieve_path_finalizes_response() -> None:
    workflow = create_workflow()

    with patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._retrieve_docs_v2', return_value={
             'standalone_question': 'What changed in OCI CLI v3 for buckets?',
             'retriever_docs': [{'page_content': 'Buckets changed in OCI CLI v3 [1].', 'metadata': {'id': 'doc-1'}}],
         }), patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._rerank_docs_v2', return_value={
             'reranker_docs': [{'page_content': 'Buckets changed in OCI CLI v3 [1].', 'metadata': {'id': 'doc-1'}}],
             'citations': [{'source': 'Doc1', 'page': 1}],
         }), patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._answer_from_docs_v2', return_value={
             'rag_answer': 'Buckets changed in OCI CLI v3 [1].',
             'citations': [{'source': 'Doc1', 'page': 1}],
             'context_usage': {'tokens': 6},
         }):
        result = workflow.invoke(
            {
                "user_request": "What changed in OCI CLI v3 for buckets?",
                "messages": [],
                "intent": "retrieve",
                "retrieval_intent": {
                    "standalone_question": "What changed in OCI CLI v3 for buckets?",
                    "search_mode": "hybrid",
                    "metadata_filters": {"product_area": "oci cli", "doc_version": "v3"},
                },
            },
            config={"configurable": {"thread_id": f"golden-retrieve-{uuid.uuid4().hex[:8]}"}},
        )

    assert result["final_answer"]
    assert result["citations"]
    assert result.get("mcp_used") is not True
    assert result.get("error") is None


def test_v2_workflow_tool_path_finalizes_protocol_and_usage() -> None:
    workflow = create_workflow()

    class FakeTool(BaseTool):
        name: str = "math.solve"
        description: str = "Solve algebra equations"

        def _run(self, *args: object, **kwargs: object) -> object:
            _ = args
            _ = kwargs
            raise NotImplementedError

    async def fake_get_mcp_tools_async(**_: object) -> list[BaseTool]:
        return [FakeTool()]

    class FakeBoundModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "math.solve",
                            "args": {"equation": "x^2 - 5x + 6 = 0"},
                            "id": "call_math_solve",
                        }
                    ],
                )
            return AIMessage(content="x = 2, 3")

    class FakeLLM:
        def bind_tools(self, tool_items: list[object], **_: object) -> FakeBoundModel:
            return FakeBoundModel()

    with patch('src.rag_agent.langgraph.nodes.run_mcp.get_mcp_tools_async', fake_get_mcp_tools_async), \
         patch('src.rag_agent.langgraph.nodes.run_mcp.get_llm', return_value=FakeLLM()), \
         patch('src.rag_agent.langgraph.nodes.run_mcp._invoke_tool_async', return_value='x = 2, 3'):
        response = workflow.invoke(
            {
                "user_request": "Solve x^2 - 5x + 6 = 0",
                "messages": [],
                "intent": "tool",
                "selected_tool_names": ["math.solve"],
                "selected_tool_descriptions": ["Solve algebra equations"],
            },
            config={"configurable": {"thread_id": f"integration-tool-{uuid.uuid4().hex[:8]}"}},
        )

    assert response["final_answer"] == "x = 2, 3"
    assert response.get("mcp_result", {}).get("status") == "success"
    assert response.get("mcp_result", {}).get("tools_used") == ["math.solve"]
    assert response.get("mcp_result", {}).get("last_tool_result") == "x = 2, 3"
    assert response.get("citations") == []


def test_v2_workflow_reformat_path_uses_existing_answer_without_retrieval() -> None:
    workflow = create_workflow()
    result = workflow.invoke(
        {
            "user_request": "Give me bullet points",
            "messages": [],
            "intent": "reformat",
            "response_instruction": "Give me bullet points",
            "final_answer": "OCI buckets are object storage containers.",
            "rag_result": {
                "status": "success",
                "answer": "OCI buckets are object storage containers.",
                "citations": [{"source": "Doc1", "page": 1}],
            },
        },
        config={"configurable": {"thread_id": f"integration-reformat-{uuid.uuid4().hex[:8]}"}},
    )

    assert result["final_answer"].startswith("- ")
    assert result["citations"] == [{"source": "Doc1", "page": 1}]


def test_v2_workflow_retrieve_path_supports_keyword_style_retrieval_intent() -> None:
    workflow = create_workflow()

    with patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._retrieve_docs_v2', return_value={
             'standalone_question': 'What does --namespace-name do in bucket create?',
             'retriever_docs': [{'page_content': '--namespace-name sets the namespace [1].', 'metadata': {'id': 'doc-1'}}],
         }), patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._rerank_docs_v2', return_value={
             'reranker_docs': [{'page_content': '--namespace-name sets the namespace [1].', 'metadata': {'id': 'doc-1'}}],
             'citations': [{'source': 'Doc1', 'page': 1}],
         }), patch('src.rag_agent.langgraph.nodes.run_rag.RunRAG._answer_from_docs_v2', return_value={
             'rag_answer': '--namespace-name sets the namespace [1].',
             'citations': [{'source': 'Doc1', 'page': 1}],
             'context_usage': {'tokens': 4},
         }):
        result = workflow.invoke(
            {
                "user_request": "What does --namespace-name do in bucket create?",
                "messages": [],
                "intent": "retrieve",
                "retrieval_intent": {
                    "standalone_question": "What does --namespace-name do in bucket create?",
                    "search_mode": "keyword",
                    "metadata_filters": {"language": "en"},
                    "top_k": 3,
                },
            },
            config={"configurable": {"thread_id": f"keyword-retrieve-{uuid.uuid4().hex[:8]}"}},
        )

    assert result["final_answer"]
    assert result["citations"]
    assert result.get("error") is None

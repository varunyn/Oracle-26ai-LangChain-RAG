from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.rag_agent.langgraph.nodes.reformat_answer import ReformatAnswer
from src.rag_agent.langgraph.nodes.run_direct import RunDirect
from src.rag_agent.langgraph.nodes.run_mcp import RunMCP
from src.rag_agent.langgraph.nodes.run_rag import RunRAG
from src.rag_agent.langgraph.retrieval_utils import STRUCTURED_FAILED_MESSAGE
from src.rag_agent.langgraph.state import DirectResult, MCPResult, MixedV2State, RAGResult


def test_run_rag_emits_success_result_without_cross_branch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunRAG()
    state: MixedV2State = {
        "user_request": "What changed in OCI CLI v3 for buckets?",
        "messages": [HumanMessage(content="What changed in OCI CLI v3 for buckets?")],
        "retrieval_intent": {
            "standalone_question": "What changed in OCI CLI v3 for buckets?",
            "search_mode": "hybrid",
            "metadata_filters": {"product_area": "oci cli", "doc_version": "v3"},
            "product_area": "oci cli",
            "doc_version": "v3",
        },
    }

    monkeypatch.setattr(node, '_retrieve_docs_v2', lambda *_args, **_kwargs: {
        'standalone_question': state['user_request'],
        'retriever_docs': [{'page_content': 'Buckets changed in OCI CLI v3 [1].', 'metadata': {'id': 'doc-1'}}],
    })
    monkeypatch.setattr(node, '_rerank_docs_v2', lambda *_args, **_kwargs: {
        'reranker_docs': [{'page_content': 'Buckets changed in OCI CLI v3 [1].', 'metadata': {'id': 'doc-1'}}],
        'citations': [{'source': 'Doc1', 'page': 1}],
    })
    monkeypatch.setattr(node, '_answer_from_docs_v2', lambda *_args, **_kwargs: {
        'rag_answer': 'Buckets changed in OCI CLI v3 [1].',
        'citations': [{'source': 'Doc1', 'page': 1}],
        'context_usage': {'tokens': 6},
    })

    result = node(state)

    rag_result = cast(RAGResult, result["rag_result"])
    assert rag_result is not None
    assert rag_result["status"] == 'success'
    assert rag_result["docs_used"] == 1
    assert rag_result["quality_score"] == 0.95
    assert rag_result["citations"]
    assert "mcp_result" not in result


def test_run_rag_returns_unavailable_when_retrieval_intent_missing() -> None:
    node = RunRAG()
    state: MixedV2State = {
        "user_request": "What changed?",
        "messages": [HumanMessage(content="What changed?")],
    }

    result = node(state)

    rag_result = cast(RAGResult, result["rag_result"])
    assert rag_result is not None
    assert rag_result["status"] == "unavailable"
    assert rag_result["docs_used"] == 0
    assert rag_result["citations"] == []


def test_run_rag_returns_ungrounded_status_without_switching_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunRAG()
    state: MixedV2State = {
        "user_request": "Tell me about that",
        "messages": [HumanMessage(content="Tell me about that")],
        "retrieval_intent": {
            "standalone_question": "Tell me about that",
            "search_mode": "semantic",
        },
    }

    monkeypatch.setattr(node, '_retrieve_docs_v2', lambda *_args, **_kwargs: {
        'standalone_question': state['user_request'],
        'retriever_docs': [{'page_content': 'Relevant source text about that.', 'metadata': {'id': 'doc-1'}}],
    })
    monkeypatch.setattr(node, '_rerank_docs_v2', lambda *_args, **_kwargs: {
        'reranker_docs': [{'page_content': 'Relevant source text about that.', 'metadata': {'id': 'doc-1'}}],
        'citations': [{'source': 'Doc1', 'page': 1}],
    })
    monkeypatch.setattr(node, '_answer_from_docs_v2', lambda *_args, **_kwargs: {
        'rag_answer': 'I found relevant documentation, but could not produce a fully grounded cited answer from the retrieved content.',
        'citations': [{'source': 'Doc1', 'page': 1}],
        'context_usage': {'tokens': 4},
    })

    result = node(state)

    rag_result = cast(RAGResult, result["rag_result"])
    assert rag_result is not None
    assert rag_result["status"] == 'ungrounded'
    assert isinstance(rag_result["citations"], list)
    assert "mcp_result" not in result


def test_run_rag_build_run_config_maps_keyword_to_text() -> None:
    node = RunRAG()
    state: MixedV2State = {"user_request": "What does --namespace-name do?", "messages": []}
    retrieval_intent = {
        "standalone_question": "What does --namespace-name do?",
        "search_mode": "keyword",
        "top_k": 7,
        "metadata_filters": {"language": "en"},
    }

    config = node._build_run_config(state, retrieval_intent)
    configurable = cast(dict[str, object], config["configurable"])

    assert configurable["search_mode"] == "text"
    assert configurable["top_k"] == 7
    assert configurable["metadata_filters"] == {"language": "en"}


def test_run_rag_retrieve_docs_v2_forwards_filters_and_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RunRAG()
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.run_rag.get_embedding_model", lambda *_args: object())

    class DummyConn:
        def __enter__(self) -> "DummyConn":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = exc_type
            _ = exc
            _ = tb

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.run_rag.get_pooled_connection", lambda: DummyConn())

    def fake_search_documents(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.run_rag.search_documents", fake_search_documents)

    result = node._retrieve_docs_v2(
        {
            "standalone_question": "What does --namespace-name do?",
            "user_request": "What does --namespace-name do?",
            "messages": [],
        },
        config={
            "configurable": {
                "collection_name": "docs",
                "search_mode": "text",
                "top_k": 4,
                "metadata_filters": {"language": "en"},
            }
        },
    )

    assert result["retriever_docs"] == []
    assert captured["top_k"] == 4
    assert captured["search_mode"] == "text"
    assert captured["metadata_filters"] == {"language": "en"}


def test_run_mcp_preserves_protocol_order_and_records_tool_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunMCP()
    state: MixedV2State = {
        "user_request": "Solve x^2 - 5x + 6 = 0",
        "messages": [HumanMessage(content="Solve x^2 - 5x + 6 = 0")],
        "selected_tool_names": ["math.solve"],
        "selected_tool_descriptions": ["Solve algebra equations"],
    }

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
        def bind_tools(self, _tool_items: list[object], **_: object) -> FakeBoundModel:
            return FakeBoundModel()

    async def fake_invoke_tool_async(*_args: object, **_kwargs: object) -> str:
        return 'x = 2, 3'

    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_mcp_tools_async', fake_get_mcp_tools_async)
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_llm', lambda: FakeLLM())
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp._invoke_tool_async', fake_invoke_tool_async)

    result = node(state)

    mcp_result = cast(MCPResult, result["mcp_result"])
    assert mcp_result is not None
    assert mcp_result["status"] == "success"
    assert mcp_result["tools_used"] == ["math.solve"]
    protocol_messages = cast(list[object], result["messages"])
    assert isinstance(protocol_messages[1], AIMessage)
    assert isinstance(protocol_messages[2], ToolMessage)
    assert isinstance(protocol_messages[3], AIMessage)
    assert cast(str, result["last_status"]) == "success"


def test_run_mcp_detects_duplicate_tool_call_and_returns_tool_failed() -> None:
    node = RunMCP()
    state: MixedV2State = {
        "user_request": "Calculate this for me",
        "messages": [
            HumanMessage(content="Calculate this for me"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "math.repeat",
                        "args": {"expression": "2 + 2"},
                        "id": "call_repeat",
                    }
                ],
            ),
            ToolMessage(content='4', tool_call_id="call_repeat", name="math.repeat"),
        ],
        "selected_tool_names": ["math.repeat"],
        "mcp_result": {
            "status": "success",
            "answer": "Existing answer",
            "tools_used": ["math.repeat"],
            "last_tool_result": "42",
        },
    }

    class FakeTool(BaseTool):
        name: str = "math.repeat"
        description: str = "Repeat test tool"

        def _run(self, *args: object, **kwargs: object) -> object:
            _ = args
            _ = kwargs
            raise NotImplementedError

    async def fake_get_mcp_tools_async(**_: object) -> list[BaseTool]:
        return [FakeTool()]

    class FakeBoundModel:
        async def ainvoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "math.repeat",
                        "args": {"expression": "2 + 2"},
                        "id": "call_repeat_again",
                    }
                ],
            )

    class FakeLLM:
        def bind_tools(self, _tool_items: list[object], **_: object) -> FakeBoundModel:
            return FakeBoundModel()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_mcp_tools_async', fake_get_mcp_tools_async)
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_llm', lambda: FakeLLM())
    try:
        result = node(state)
    finally:
        monkeypatch.undo()

    mcp_result = cast(MCPResult, result["mcp_result"])
    assert mcp_result is not None
    assert mcp_result["status"] == "tool_failed"
    assert mcp_result["tools_used"] == ["math.repeat"]
    assert mcp_result["last_tool_result"] is None
    assert "Tool 'math.repeat' failed" in mcp_result["answer"]
    assert cast(str, result["last_status"]) == "tool_failed"


def test_run_mcp_returns_provider_failed_when_no_tools_selected() -> None:
    node = RunMCP()
    state: MixedV2State = {
        "user_request": "Compute this",
        "messages": [HumanMessage(content="Compute this")],
        "selected_tool_names": [],
    }

    result = node(state)

    mcp_result = cast(MCPResult, result["mcp_result"])
    assert mcp_result is not None
    assert mcp_result["status"] == "provider_failed"
    assert mcp_result["tools_used"] == []


def test_run_direct_emits_direct_result() -> None:
    node = RunDirect()
    state: MixedV2State = {
        "user_request": "Hello there",
        "messages": [HumanMessage(content="Hello there")],
    }

    result = node(state)

    direct_result = cast(DirectResult, result["direct_result"])
    assert direct_result is not None
    assert direct_result["status"] == "success"
    assert direct_result["answer"] == "Hello there"
    assert result["last_status"] == "success"


def test_reformat_answer_uses_prior_grounded_answer_without_retrieval() -> None:
    node = ReformatAnswer()
    state: MixedV2State = {
        "user_request": "Give me bullet points",
        "messages": [HumanMessage(content="Give me bullet points")],
        "response_instruction": "Give me bullet points",
        "final_answer": "OCI buckets are object storage containers.",
        "rag_result": {
            "status": "success",
            "answer": "OCI buckets are object storage containers.",
            "citations": [{"source": "Doc1", "page": 1}],
        },
    }

    result = node(state)

    assert cast(str, result["final_answer"]).startswith("- ")
    assert result["citations"] == [{"source": "Doc1", "page": 1}]
    assert "rag_result" not in result
    assert "mcp_result" not in result


def test_run_rag_maps_legacy_structured_failure_to_v2_ungrounded_message(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunRAG()
    state: MixedV2State = {
        "user_request": "How do I import an existing visual application?",
        "messages": [HumanMessage(content="How do I import an existing visual application?")],
        "retrieval_intent": {
            "standalone_question": "How do I import an existing visual application?",
            "search_mode": "semantic",
        },
    }

    monkeypatch.setattr(node, '_retrieve_docs_v2', lambda *_args, **_kwargs: {
        'standalone_question': state['user_request'],
        'retriever_docs': [
            {
                'page_content': 'Click Import, then Application from file, then upload the archive.',
                'metadata': {'id': 'doc-1'},
            }
        ],
    })
    monkeypatch.setattr(node, '_rerank_docs_v2', lambda *_args, **_kwargs: {
        'reranker_docs': [
            {
                'page_content': 'Click Import, then Application from file, then upload the archive.',
                'metadata': {'id': 'doc-1'},
            }
        ],
        'citations': [{'source': 'Doc1', 'page': 1}],
    })
    monkeypatch.setattr(
        node,
        '_answer_from_docs_v2',
        lambda *_args, **_kwargs: {
            'rag_answer': STRUCTURED_FAILED_MESSAGE,
            'citations': [{'source': 'Doc1', 'page': 1}],
            'context_usage': {'tokens': 10},
            'rag_has_citations': False,
        },
    )

    result = node(state)

    rag_result = cast(RAGResult, result['rag_result'])
    assert rag_result['status'] == 'ungrounded'
    assert rag_result['answer'] != STRUCTURED_FAILED_MESSAGE
    assert "couldn't generate a cited answer" not in rag_result["answer"].lower()
    assert result['last_status'] == 'ungrounded'


def test_finalize_response_retrieve_uses_v2_rag_answer_for_ungrounded_result() -> None:
    from src.rag_agent.langgraph.nodes.finalize_response import FinalizeResponse

    state: MixedV2State = {
        'user_request': 'How do I import an existing visual application?',
        'messages': [HumanMessage(content='How do I import an existing visual application?')],
        'intent': 'retrieve',
        'rag_result': {
                'status': 'ungrounded',
                'answer': 'I found relevant documentation, but could not produce a fully grounded cited answer from the retrieved content.',
                'citations': [{'source': 'Doc1', 'page': 1}],
                'docs_used': 1,
                'quality_score': 0.35,
            },
            'reranker_docs': [{'page_content': 'doc', 'metadata': {'id': 'd1'}}],
            'context_usage': {'tokens': 10},
    }

    result = FinalizeResponse()(state)

    assert result['final_answer'] == 'I found relevant documentation, but could not produce a fully grounded cited answer from the retrieved content.'
    assert result['error'] is None
    assert result['citations'] == [{'source': 'Doc1', 'page': 1}]


def test_run_rag_uses_v2_grounding_even_when_legacy_rag_has_citations_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RunRAG()
    state: MixedV2State = {
        'user_request': 'How do I import an existing visual application?',
        'messages': [HumanMessage(content='How do I import an existing visual application?')],
        'retrieval_intent': {
            'standalone_question': 'How do I import an existing visual application?',
            'search_mode': 'semantic',
        },
    }

    grounded_answer = 'To import an existing visual application, click Import and choose Application from file [1].'
    grounded_doc = {
        'page_content': 'To import an existing visual application, click Import and choose Application from file.',
        'metadata': {'id': 'doc-1'},
    }

    monkeypatch.setattr(node, '_retrieve_docs_v2', lambda *_args, **_kwargs: {
        'standalone_question': state['user_request'],
        'retriever_docs': [grounded_doc],
    })
    monkeypatch.setattr(node, '_rerank_docs_v2', lambda *_args, **_kwargs: {
        'reranker_docs': [grounded_doc],
        'citations': [{'source': 'Doc1', 'page': 1}],
    })
    monkeypatch.setattr(
        node,
        '_answer_from_docs_v2',
        lambda *_args, **_kwargs: {
            'rag_answer': grounded_answer,
            'citations': [{'source': 'Doc1', 'page': 1}],
            'context_usage': {'tokens': 8},
            'rag_has_citations': False,
        },
    )

    result = node(state)

    rag_result = cast(RAGResult, result['rag_result'])
    assert rag_result['status'] == 'success'
    assert rag_result['answer'] == grounded_answer
    assert rag_result['quality_score'] == 0.95


def test_run_rag_uses_v2_answer_node_instead_of_legacy_answer_from_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunRAG()
    state: MixedV2State = {
        'user_request': 'How do I import an existing visual application?',
        'messages': [HumanMessage(content='How do I import an existing visual application?')],
        'retrieval_intent': {
            'standalone_question': 'How do I import an existing visual application?',
            'search_mode': 'semantic',
        },
    }

    monkeypatch.setattr(node, '_retrieve_docs_v2', lambda *_args, **_kwargs: {
        'standalone_question': state['user_request'],
        'retriever_docs': [
            {
                'page_content': 'To import an existing visual application, click Import and choose Application from file.',
                'metadata': {'id': 'doc-1'},
            }
        ],
    })
    monkeypatch.setattr(node, '_rerank_docs_v2', lambda *_args, **_kwargs: {
        'reranker_docs': [
            {
                'page_content': 'To import an existing visual application, click Import and choose Application from file.',
                'metadata': {'id': 'doc-1'},
            }
        ],
        'citations': [{'source': 'Doc1', 'page': 1}],
    })

    monkeypatch.setattr(
        node,
        '_answer_from_docs_v2',
        lambda *_args, **_kwargs: {
            'rag_answer': 'To import an existing visual application, click Import and choose Application from file [1].',
            'citations': [{'source': 'Doc1', 'page': 1}],
            'context_usage': {'tokens': 12},
        },
    )

    result = node(state)

    rag_result = cast(RAGResult, result['rag_result'])
    assert rag_result['status'] == 'success'
    assert 'Application from file [1]' in rag_result['answer']


def test_run_rag_uses_v2_search_and_rerank_seams_instead_of_legacy_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RunRAG()
    state: MixedV2State = {
        'user_request': 'What changed in OCI CLI v3 for buckets?',
        'messages': [HumanMessage(content='What changed in OCI CLI v3 for buckets?')],
        'retrieval_intent': {
            'standalone_question': 'What changed in OCI CLI v3 for buckets?',
            'search_mode': 'hybrid',
        },
    }

    def fail_if_legacy_search_used(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError('RunRAG should not invoke legacy SemanticSearch')

    def fail_if_legacy_reranker_used(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError('RunRAG should not invoke legacy Reranker')

    monkeypatch.setattr(node, '_retrieve_docs_v2', fail_if_legacy_search_used)
    monkeypatch.setattr(node, '_rerank_docs_v2', fail_if_legacy_reranker_used)
    monkeypatch.setattr(
        node,
        '_retrieve_docs_v2',
        lambda *_args, **_kwargs: {
            'standalone_question': 'What changed in OCI CLI v3 for buckets?',
            'retriever_docs': [
                {
                    'page_content': 'Buckets changed in OCI CLI v3 [1].',
                    'metadata': {'source': 'Doc1', 'page': 1},
                }
            ],
        },
    )
    monkeypatch.setattr(
        node,
        '_rerank_docs_v2',
        lambda *_args, **_kwargs: {
            'reranker_docs': [
                {
                    'page_content': 'Buckets changed in OCI CLI v3 [1].',
                    'metadata': {'source': 'Doc1', 'page': 1},
                }
            ],
            'citations': [{'source': 'Doc1', 'page': 1}],
        },
    )
    monkeypatch.setattr(
        node,
        '_answer_from_docs_v2',
        lambda *_args, **_kwargs: {
            'rag_answer': 'Buckets changed in OCI CLI v3 [1].',
            'citations': [{'source': 'Doc1', 'page': 1}],
            'context_usage': {'tokens': 7},
        },
    )

    result = node(state)

    rag_result = cast(RAGResult, result['rag_result'])
    assert rag_result['status'] == 'success'
    assert rag_result['citations'] == [{'source': 'Doc1', 'page': 1}]


def test_run_mcp_allows_same_tool_on_new_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RunMCP()
    state: MixedV2State = {
        "user_request": "Solve a new equation",
        "messages": [
            HumanMessage(content="Calculate the integral of x^2 * e^x."),
            AIMessage(content="", tool_calls=[{"name": "calculator_integrate", "args": {"expression": "x**2 * exp(x)", "variable": "x"}, "id": "call_integrate"}]),
            ToolMessage(content='{"result": "(x**2 - 2*x + 2)*exp(x)"}', tool_call_id="call_integrate", name="calculator_integrate"),
            AIMessage(content="The integral is (x^2 - 2*x + 2)*e^x."),
            HumanMessage(content="Solve a new equation"),
        ],
        "selected_tool_names": ["calculator_integrate"],
        "selected_tool_descriptions": ["Integrates mathematical expressions"],
        "mcp_result": {
            "status": "success",
            "answer": "The integral is (x^2 - 2*x + 2)*e^x.",
            "tools_used": ["calculator_integrate"],
            "last_tool_result": '(x**2 - 2*x + 2)*exp(x)',
        },
    }

    class FakeTool(BaseTool):
        name: str = "calculator_integrate"
        description: str = "Integrates mathematical expressions"

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
                            "name": "calculator_integrate",
                            "args": {"expression": "x**2", "variable": "x"},
                            "id": "call_integrate_again",
                        }
                    ],
                )
            return AIMessage(content="The integral of x^2 is x^3/3.")

    class FakeLLM:
        def bind_tools(self, _tool_items: list[object], **_: object) -> FakeBoundModel:
            return FakeBoundModel()

        async def ainvoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return AIMessage(content="The integral of x^2 is x^3/3.")

    async def fake_invoke_tool_async(*_args: object, **_kwargs: object) -> str:
        return 'x**3/3'

    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_mcp_tools_async', fake_get_mcp_tools_async)
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp.get_llm', lambda: FakeLLM())
    monkeypatch.setattr('src.rag_agent.langgraph.nodes.run_mcp._invoke_tool_async', fake_invoke_tool_async)

    result = node(state)

    mcp_result = cast(MCPResult, result['mcp_result'])
    assert mcp_result['status'] == 'success'
    assert mcp_result['tools_used'] == ['calculator_integrate']
    assert mcp_result['answer'] == 'The integral of x^2 is x^3/3.'

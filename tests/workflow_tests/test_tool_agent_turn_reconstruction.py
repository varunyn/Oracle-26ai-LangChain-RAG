from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from src.rag_agent.graphs import tool_agent_execution, tool_agent_turn
from src.rag_agent.graphs.chat_agent import build_chat_agent
from src.rag_agent.graphs.mcp_policies import (
    NO_ORACLE_CONTEXT_ANSWER,
    ORACLE_RETRIEVAL_FAILED_ANSWER,
)
from src.rag_agent.graphs.nodes import mixed
from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.graphs.tool_agent_execution import build_tool_agent_sub_graph
from src.rag_agent.graphs.tool_agent_turn import IncompatibleMCPConfigurationError
from src.rag_agent.runtime.agent_server_checkpointer import LocalAsyncSqliteSaver
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore
from src.rag_agent.runtime.tool_agent_recipe_store import (
    MissingRecipeError,
    RecipeConflictError,
    ToolAgentTurnRecipe,
)


def test_mcp_setup_interrupt_resume_uses_stored_selections_without_leaking_recipe(
    tmp_path, monkeypatch
) -> None:
    sentinel = "recipe-sentinel-model"

    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return []

    class FakeModel:
        model_id = sentinel

        async def ainvoke(self, _messages: object, *, config: object) -> AIMessage:
            return AIMessage(id="ai-final", content="done")

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(tool_agent_execution, "get_llm", lambda **_kwargs: FakeModel())

    async def run() -> tuple[dict, dict, bytes]:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "resume.sqlite")) as saver:
            graph = build_chat_agent(checkpointer=saver)
            config = {"configurable": {"thread_id": "resume-thread", "run_id": "origin"}}
            context = {
                "mode": "mcp",
                "model_id": sentinel,
                "mcp_server_keys": [],
                "session_id": "session-origin",
            }
            interrupted = await graph.ainvoke(
                {"messages": [HumanMessage(id="human-1", content="hello")]},
                config,
                context=context,
                interrupt_after=["mcp_setup"],
            )
            resumed = await graph.ainvoke(
                None,
                {"configurable": {"thread_id": "resume-thread", "run_id": "continuation"}},
                context={
                    "mode": "mcp",
                    "model_id": "changed-request-default",
                    "mcp_server_keys": [],
                    "session_id": "changed-session",
                },
            )
            async with saver.conn.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ?", ("resume-thread",)
            ) as cursor:
                raw = b"".join(bytes(row[0]) for row in await cursor.fetchall())
            return interrupted, resumed, raw

    interrupted, resumed, raw = asyncio.run(run())
    assert interrupted["messages"][-1].id == "human-1"
    assert resumed["messages"][-1].content == "done"
    assert all(message.id != sentinel for message in resumed["messages"])
    assert sentinel.encode() not in raw


def test_tool_loop_interrupt_resume_reconstructs_original_model_and_message_ids(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def lookup(value: str) -> str:
        return f"looked-up:{value}"

    lookup_tool = StructuredTool.from_function(lookup, name="lookup", description="lookup")

    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return [lookup_tool]

    class FakeModel:
        model_id = "original-model"

        def bind_tools(self, _tools: list[object]) -> FakeModel:
            return self

        async def ainvoke(self, _messages: object, *, config: object) -> AIMessage:
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(
                    id="ai-tool-call",
                    content="",
                    tool_calls=[{"id": "tool-call-1", "name": "lookup", "args": {"value": "x"}}],
                )
            return AIMessage(id="ai-answer", content="finished")

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(tool_agent_execution, "get_llm", lambda **_kwargs: FakeModel())

    async def run() -> dict:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "loop.sqlite")) as saver:
            graph = build_tool_agent_sub_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "loop-thread", "run_id": "run-loop"}}
            context = {"mode": "mcp", "model_id": "original-model", "mcp_server_keys": []}
            setup_runtime = type(
                "Runtime",
                (),
                {
                    "context": context,
                    "execution_info": type(
                        "Info", (), {"thread_id": "loop-thread", "run_id": "run-loop"}
                    )(),
                },
            )()
            setup_turn = await tool_agent_turn.prepare_tool_agent_turn(
                state={"messages": [HumanMessage(id="human-loop", content="look up x")]},
                parent_config={"configurable": {"__pregel_checkpointer": saver}},
                runtime=setup_runtime,
                mode="mcp",
            )
            await tool_agent_turn.release_tool_agent_turn(
                {"configurable": {"__pregel_checkpointer": saver}}, setup_turn
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(id="human-loop", content="look up x")]},
                config,
                context=context,
                interrupt_after=["call_llm"],
            )
            return await graph.ainvoke(
                None,
                config,
                context={"mode": "mcp", "model_id": "changed-model", "mcp_server_keys": []},
            )

    result = asyncio.run(run())
    assert calls == 2
    assert [message.id for message in result["messages"]][0:2] == [
        "human-loop",
        "ai-tool-call",
    ]
    assert result["messages"][2].tool_call_id == "tool-call-1"
    assert result["messages"][3].id == "ai-answer"


def test_resume_rejects_incompatible_stored_mcp_configuration(tmp_path, monkeypatch) -> None:
    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    from src.rag_agent.infrastructure import mcp_settings

    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: {})

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "mcp-drift.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {
                        "model_id": "m",
                        "mcp_server_keys": ["s"],
                        "mcp_config_digest": "v1",
                    },
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r1"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            await saver.recipe_store.create_or_load(
                ToolAgentTurnRecipe(
                    thread_id="t",
                    turn_id="h",
                    origin_run_id="r1",
                    request_id=None,
                    session_id=None,
                    mode="mcp",
                    model_key="m",
                    collection_key=None,
                    mcp_server_keys=("s",),
                    mcp_config_digest="matching-supplied-digest",
                    enable_tracing=False,
                    tool_round_limit=4,
                )
            )
            runtime.context["mcp_config_digest"] = "matching-supplied-digest"
            with pytest.raises(IncompatibleMCPConfigurationError):
                await tool_agent_turn.reconstruct_tool_agent_turn(
                    state=state, parent_config=config, runtime=runtime, mode="mcp"
                )

    asyncio.run(run())


def test_resume_rejects_live_mcp_definition_drift(tmp_path, monkeypatch) -> None:
    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    from src.rag_agent.infrastructure import mcp_settings

    definitions = {"s": {"transport": "streamable-http", "url": "https://one"}}
    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: definitions)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "live-mcp-drift.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": ["s"]},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r1"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            turn = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, turn)
            definitions["s"] = {"transport": "streamable-http", "url": "https://two"}
            with pytest.raises(IncompatibleMCPConfigurationError):
                await tool_agent_turn.reconstruct_tool_agent_turn(
                    state=state, parent_config=config, runtime=runtime, mode="mcp"
                )

    asyncio.run(run())


def test_resume_accepts_mcp_credential_and_header_rotation(tmp_path, monkeypatch) -> None:
    async def no_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_tools)
    from src.rag_agent.infrastructure import mcp_settings

    definitions = {
        "s": {
            "transport": "streamable-http",
            "url": "https://one",
            "auth": {"type": "bearer", "bearer_token": "token-one"},
            "headers": {"Authorization": "Bearer token-one"},
        }
    }
    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: definitions)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "credential-rotation.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": ["s"]},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            setup = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, setup)

            definitions["s"]["auth"] = {
                "type": "bearer",
                "bearer_token": "token-two",
            }
            definitions["s"]["headers"] = {
                "Authorization": "Bearer token-two",
            }
            resumed = await tool_agent_turn.reconstruct_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, resumed)

    asyncio.run(run())


def test_implicit_mcp_selection_persists_all_keys_and_rejects_deletion(
    tmp_path, monkeypatch
) -> None:
    async def no_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_tools)
    from src.rag_agent.infrastructure import mcp_settings

    definitions = {
        "alpha": {"transport": "streamable-http", "url": "https://alpha"},
        "beta": {"transport": "streamable-http", "url": "https://beta"},
    }
    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: definitions)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "implicit-mcp.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m"},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r1"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            turn = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, turn)
            recipe = await saver.recipe_store.load(("t", "h"))
            assert recipe is not None
            assert recipe.mcp_server_keys == ("alpha", "beta")
            del definitions["beta"]
            with pytest.raises(IncompatibleMCPConfigurationError):
                await tool_agent_turn.reconstruct_tool_agent_turn(
                    state=state, parent_config=config, runtime=runtime, mode="mcp"
                )

    asyncio.run(run())


def test_load_only_reconstruction_does_not_evaluate_changed_defaults(tmp_path, monkeypatch) -> None:
    async def no_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_tools)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "load-only.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "original", "mcp_server_keys": []},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r1"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            setup = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, setup)

            def unavailable(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("load-only reconstruction evaluated a mutable default")

            monkeypatch.setattr(tool_agent_turn, "get_llm", unavailable)
            monkeypatch.setattr(tool_agent_turn, "get_settings", unavailable)
            monkeypatch.setattr(
                tool_agent_turn,
                "build_run_config",
                lambda **_kwargs: {"configurable": {"mode": "mcp"}},
            )
            resumed = await tool_agent_turn.reconstruct_tool_agent_turn(
                state=state,
                parent_config=config,
                runtime=runtime,
                mode="mcp",
            )
            assert resumed["model_id"] == "original"
            await tool_agent_turn.release_tool_agent_turn(config, resumed)

    asyncio.run(run())


def test_setup_failure_replay_under_new_run_preserves_origin_and_links_run(
    tmp_path, monkeypatch
) -> None:
    attempts = 0

    async def fail_once_then_load(**_kwargs: object) -> list[object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("crash before setup checkpoint")
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fail_once_then_load)

    async def run() -> tuple[object, list[tuple[str, str]]]:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "new-run-replay.sqlite")
        ) as saver:
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="question")]}
            first_runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": []},
                    "execution_info": type(
                        "Info", (), {"thread_id": "t", "run_id": "origin-run"}
                    )(),
                },
            )()
            with pytest.raises(RuntimeError, match="crash before setup"):
                await tool_agent_turn.prepare_tool_agent_turn(
                    state=state, parent_config=config, runtime=first_runtime, mode="mcp"
                )

            replay_runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": []},
                    "execution_info": type(
                        "Info", (), {"thread_id": "t", "run_id": "continuation-run"}
                    )(),
                },
            )()
            turn = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=replay_runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, turn)
            recipe = await saver.recipe_store.load(("t", "h"))
            assert recipe is not None
            async with saver.conn.execute(
                "SELECT run_id, turn_id FROM tool_agent_turn_run_links WHERE thread_id = ? ORDER BY run_id",
                ("t",),
            ) as cursor:
                links = await cursor.fetchall()
            return recipe, [(str(run_id), str(turn_id)) for run_id, turn_id in links]

    recipe, links = asyncio.run(run())
    assert recipe.origin_run_id == "origin-run"
    assert links == [("continuation-run", "h"), ("origin-run", "h")]


def test_explicit_empty_selection_cannot_gain_tools_from_current_config(
    tmp_path, monkeypatch
) -> None:
    definitions = {"alpha": {"transport": "streamable-http", "url": "https://alpha"}}
    observed: list[object] = []
    from src.rag_agent.infrastructure import mcp_settings

    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: definitions)

    async def no_tools(**kwargs: object) -> list[object]:
        observed.append(kwargs["server_keys"])
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_tools)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "empty-selection.sqlite")
        ) as saver:
            config = {"configurable": {"__pregel_checkpointer": saver}}
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": []},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r"})(),
                },
            )()
            state = {"messages": [HumanMessage(id="h", content="question")]}
            setup = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, setup)
            runtime.context["mcp_server_keys"] = ["alpha"]
            resumed = await tool_agent_turn.reconstruct_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, resumed)

    asyncio.run(run())
    assert observed == [
        [tool_agent_turn._EMPTY_MCP_SELECTION],
        [tool_agent_turn._EMPTY_MCP_SELECTION],
    ]


def test_missing_explicit_mcp_key_is_rejected_before_recipe_creation(tmp_path, monkeypatch) -> None:
    from src.rag_agent.infrastructure import mcp_settings

    monkeypatch.setattr(mcp_settings, "get_mcp_servers_config", lambda: {})

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "missing-selection.sqlite")
        ) as saver:
            config = {"configurable": {"__pregel_checkpointer": saver}}
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "m", "mcp_server_keys": ["missing"]},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r"})(),
                },
            )()
            with pytest.raises(IncompatibleMCPConfigurationError):
                await tool_agent_turn.prepare_tool_agent_turn(
                    state={"messages": [HumanMessage(id="h", content="question")]},
                    parent_config=config,
                    runtime=runtime,
                    mode="mcp",
                )
            assert await saver.recipe_store.load(("t", "h")) is None

    asyncio.run(run())


def test_checkpointed_retrieval_failure_remains_distinct_from_empty_context(
    tmp_path, monkeypatch
) -> None:
    async def no_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_tools)

    async def run_case(db_name: str, artifact: list[object]) -> str:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / db_name)) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"mode": "mixed", "model_id": "m", "mcp_server_keys": []},
                    "execution_info": type("Info", (), {"thread_id": db_name, "run_id": "r"})(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            state = {"messages": [HumanMessage(id="h", content="find context")]}
            setup = await mixed.run_mixed_mcp_setup(state, config, runtime)
            assert setup["progress"]

            def emit_messages(_state: ChatGraphState) -> ChatGraphState:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "retrieval-call",
                                    "name": "oracle_retrieval",
                                    "args": {"query": "find context"},
                                }
                            ],
                        ),
                        ToolMessage(
                            content="retrieval result",
                            name="oracle_retrieval",
                            tool_call_id="retrieval-call",
                            artifact=artifact,
                        ),
                    ]
                }

            graph_builder = StateGraph(ChatGraphState)
            graph_builder.add_node("emit", emit_messages)
            graph_builder.add_edge(START, "emit")
            graph_builder.add_edge("emit", END)
            graph = graph_builder.compile(checkpointer=saver)
            await graph.ainvoke(state, {"configurable": {"thread_id": db_name}})
            checkpoint = await graph.aget_state({"configurable": {"thread_id": db_name}})
            result = await mixed.run_mixed_compose_node(
                {"messages": checkpoint.values["messages"]}, config, runtime
            )
            return str(result["messages"][-1].content)

    failure = asyncio.run(
        run_case("failure.sqlite", [{"type": "oracle_retrieval_error", "error": "db down"}])
    )
    empty = asyncio.run(run_case("empty.sqlite", []))
    assert failure == ORACLE_RETRIEVAL_FAILED_ANSWER
    assert empty == NO_ORACLE_CONTEXT_ANSWER


def test_mixed_compose_reconstructs_real_retrieval_artifact(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    document = Document(page_content="durable evidence", metadata={"source": "spec"})

    async def no_mcp_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_mcp_tools)

    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda question, docs, **_kwargs: observed.update({"docs": docs}) or docs,
    )
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [])
    monkeypatch.setattr(
        mixed.rag_runtime,
        "synthesize_rag_answer",
        lambda **_kwargs: _synthesized_answer(observed, _kwargs),
    )

    async def run() -> dict:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "mixed.sqlite")) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {
                        "mode": "mixed",
                        "model_id": "model",
                        "collection_name": "ORIGINAL_COLLECTION",
                        "mcp_server_keys": [],
                    },
                    "execution_info": type(
                        "Info", (), {"thread_id": "mixed-thread", "run_id": "run"}
                    )(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            messages = [
                HumanMessage(id="human-mixed", content="find evidence"),
                AIMessage(
                    id="ai-tool",
                    content="",
                    tool_calls=[
                        {
                            "id": "oracle-call",
                            "name": "oracle_retrieval",
                            "args": {"query": "find evidence"},
                        }
                    ],
                ),
                ToolMessage(
                    id="tool-result",
                    content="durable evidence",
                    name="oracle_retrieval",
                    tool_call_id="oracle-call",
                    artifact=[document],
                ),
            ]
            setup = await mixed.run_mixed_mcp_setup({"messages": [messages[0]]}, config, runtime)
            assert setup["progress"]
            runtime.context["model_id"] = "changed-model"
            return await mixed.run_mixed_compose_node({"messages": messages}, config, runtime)

    result = asyncio.run(run())
    assert observed["docs"] == [document]
    assert observed["synthesis_model"] == "model"
    assert result["references"]["context_usage"] == {"retrieved_docs_count": 1}


def test_mixed_resume_preserves_turn_configuration_and_current_turn_evidence(
    tmp_path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    old_document = Document(page_content="old evidence", metadata={"source": "old"})
    current_document = Document(page_content="current evidence", metadata={"source": "current"})

    async def no_mcp_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_mcp_tools)
    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda question, docs, **kwargs: observed.update(
            {
                "rerank_question": question,
                "rerank_docs": docs,
                "rerank_enabled": kwargs["enable_reranker"],
            }
        )
        or docs,
    )
    monkeypatch.setattr(
        mixed.rag_runtime,
        "citations_from_docs",
        lambda docs: [{"source": doc.metadata["source"]} for doc in docs],
    )
    monkeypatch.setattr(
        mixed.rag_runtime,
        "serialize_docs",
        lambda docs: [{"source": doc.metadata["source"]} for doc in docs],
    )

    async def synthesize(**kwargs: object) -> tuple[str, None, str]:
        observed.update(
            {
                "synthesis_model": kwargs["model_id"],
                "synthesis_docs": kwargs["docs"],
                "synthesis_run_config": kwargs["run_config"],
            }
        )
        return "replayed answer", None, "original-model"

    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", synthesize)

    async def run() -> dict:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "mixed-fidelity.sqlite")
        ) as saver:
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {
                        "mode": "mixed",
                        "model_id": "original-model",
                        "collection_name": "ORIGINAL_COLLECTION",
                        "enable_reranker": True,
                        "mcp_server_keys": [],
                    },
                    "execution_info": type(
                        "Info", (), {"thread_id": "mixed-thread", "run_id": "origin"}
                    )(),
                },
            )()
            config = {"configurable": {"__pregel_checkpointer": saver}}
            setup = await mixed.run_mixed_mcp_setup(
                {"messages": [HumanMessage(id="current-human", content="find current")]},
                config,
                runtime,
            )
            assert setup["progress"]
            messages = [
                HumanMessage(id="old-human", content="find old"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "old-call", "name": "oracle_retrieval", "args": {"query": "old"}}
                    ],
                ),
                ToolMessage(
                    content="old evidence",
                    name="oracle_retrieval",
                    tool_call_id="old-call",
                    artifact=[old_document],
                ),
                HumanMessage(id="current-human", content="find current"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "current-call",
                            "name": "oracle_retrieval",
                            "args": {"query": "find current"},
                        }
                    ],
                ),
                ToolMessage(
                    content="current evidence",
                    name="oracle_retrieval",
                    tool_call_id="current-call",
                    artifact=[current_document],
                ),
            ]
            runtime.context.update(
                {
                    "model_id": "changed-model",
                    "collection_name": "CHANGED_COLLECTION",
                    "enable_reranker": False,
                }
            )
            return await mixed.run_mixed_compose_node({"messages": messages}, config, runtime)

    result = asyncio.run(run())
    assert observed["rerank_enabled"] is True
    assert observed["rerank_docs"] == [current_document]
    assert observed["synthesis_model"] == "original-model"
    assert observed["synthesis_docs"] == [current_document]
    assert result["references"]["citations"] == [{"source": "current"}]
    assert result["references"]["reranker_docs"] == [{"source": "current"}]


async def _synthesized_answer(observed: dict[str, object], kwargs: dict[str, object]):
    observed["synthesis_docs"] = kwargs["docs"]
    observed["synthesis_model"] = kwargs["model_id"]
    return "answer", None, "model"


def test_setup_replays_call_create_or_load_but_reconstruction_requires_recipe(
    tmp_path, monkeypatch
) -> None:
    async def no_mcp_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_mcp_tools)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "identity.sqlite")
        ) as saver:
            config = {"configurable": {"__pregel_checkpointer": saver}}
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {"model_id": "original", "mcp_server_keys": []},
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r"})(),
                },
            )()
            state = {"messages": [HumanMessage(id="h", content="question")]}
            first = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, first)
            runtime.context["model_id"] = "changed"
            with pytest.raises(RecipeConflictError):
                await tool_agent_turn.prepare_tool_agent_turn(
                    state=state, parent_config=config, runtime=runtime, mode="mcp"
                )

            async with LocalAsyncSqliteSaver.from_conn_string(
                str(tmp_path / "missing.sqlite")
            ) as empty:
                missing_config = {"configurable": {"__pregel_checkpointer": empty}}
                with pytest.raises(MissingRecipeError):
                    await tool_agent_turn.reconstruct_tool_agent_turn(
                        state=state, parent_config=missing_config, runtime=runtime, mode="mcp"
                    )

    asyncio.run(run())


def test_reconstruction_pins_recipe_mode_and_round_limit(tmp_path, monkeypatch) -> None:
    async def no_mcp_tools(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", no_mcp_tools)

    async def run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "pinned.sqlite")) as saver:
            config = {"configurable": {"__pregel_checkpointer": saver}}
            runtime = type(
                "Runtime",
                (),
                {
                    "context": {
                        "mode": "mcp",
                        "model_id": "original",
                        "mcp_server_keys": [],
                        "max_rounds": 7,
                    },
                    "execution_info": type("Info", (), {"thread_id": "t", "run_id": "r"})(),
                },
            )()
            state = {"messages": [HumanMessage(id="h", content="question")]}
            setup = await tool_agent_turn.prepare_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mcp"
            )
            await tool_agent_turn.release_tool_agent_turn(config, setup)
            runtime.context.update({"mode": "mixed", "model_id": "changed", "max_rounds": 1})
            turn = await tool_agent_turn.reconstruct_tool_agent_turn(
                state=state, parent_config=config, runtime=runtime, mode="mixed"
            )
            try:
                assert turn["tool_round_limit"] == 7
                assert turn["run_config"]["configurable"]["mode"] == "mcp"
            finally:
                await tool_agent_turn.release_tool_agent_turn(config, turn)

    asyncio.run(run())


def test_persisted_evidence_normalizes_documents_and_starts_after_latest_human() -> None:
    evidence = OracleRetrievalEvidenceStore.from_persisted_messages(
        [
            HumanMessage(id="old-human", content="old"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "old-call", "name": "oracle_retrieval", "args": {"query": "old"}}
                ],
            ),
            ToolMessage(
                name="oracle_retrieval",
                tool_call_id="old-call",
                artifact=[{"page_content": "old doc", "metadata": {"source": "old"}}],
            ),
            HumanMessage(id="new-human", content="new"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "new-call", "name": "oracle_retrieval", "args": {"query": "new"}}
                ],
            ),
            ToolMessage(
                name="oracle_retrieval",
                tool_call_id="new-call",
                artifact=[
                    {
                        "type": "Document",
                        "data": {"page_content": "new doc", "metadata": {"source": "new"}},
                    }
                ],
            ),
        ],
        collection_name="collection",
    )
    selected = evidence.read()
    assert selected is not None
    assert [document.page_content for document in selected.documents] == ["new doc"]
    assert selected.documents[0].metadata == {"source": "new"}


def test_checkpoint_reload_evidence_is_limited_to_latest_turn(tmp_path) -> None:
    document = Document(page_content="checkpointed new", metadata={"source": "new"})

    def append_retrieval(_state: ChatGraphState) -> ChatGraphState:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "checkpoint-call",
                            "name": "oracle_retrieval",
                            "args": {"query": "new"},
                        }
                    ],
                ),
                ToolMessage(
                    content="checkpointed new",
                    name="oracle_retrieval",
                    tool_call_id="checkpoint-call",
                    artifact=[document],
                ),
            ]
        }

    async def run() -> OracleRetrievalEvidenceStore:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "evidence.sqlite")
        ) as saver:
            builder = StateGraph(ChatGraphState)
            builder.add_node("append", append_retrieval)
            builder.add_edge(START, "append")
            builder.add_edge("append", END)
            graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "evidence-thread"}}
            await graph.ainvoke(
                {"messages": [HumanMessage(id="old-human", content="old turn")]}, config
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(id="new-human", content="new turn")]}, config
            )
            checkpoint = await graph.aget_state(config)
            return OracleRetrievalEvidenceStore.from_persisted_messages(
                checkpoint.values["messages"], collection_name="collection"
            )

    selected = asyncio.run(run()).read()
    assert selected is not None
    assert [item.page_content for item in selected.documents] == ["checkpointed new"]

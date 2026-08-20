# Tool-Agent Turn Phase 1 Plan

## Status

Implemented on 2026-08-20; no commit created.

## Goal

Deepen the shared MCP and mixed-mode tool-agent execution module without changing the browser-visible tool-progress stream, persisted thread history, graph node names, or final chat result contract.

## Decisions recorded

- The frontend continues to render live tool activity from the existing streamed AI and tool messages. A turn that makes ten tool calls must continue to show those calls as they happen.
- Persisted messages remain the source for thread replay and recovery.
- `ToolExecutionTranscript` is a transient, post-execution normalization of the completed message sequence. It must not be added to the frontend stream or persisted thread contract.
- Shared LLM/tool-loop mechanics move out of `src/rag_agent/graphs/nodes/mixed.py` into `src/rag_agent/graphs/tool_agent_execution.py`.
- MCP-specific and mixed-specific outcome policy remain in their respective node modules.
- The mutable `runtime.context["tool_agent_turn"]` handoff remains unchanged in phase 1. Replacing it with typed private graph state is phase 2.
- Phase 1 began with deterministic unit and workflow coverage; live Agent Server validation was subsequently added to prove the stream and persistence contracts against the real MCP path.

## Target ownership

```text
tool_agent_turn.py
  prepares the per-request ToolAgentTurn

tool_agent_execution.py
  shared LLM/tool loop, provider message normalization, loop routing,
  and post-execution ToolExecutionTranscript derivation

nodes/mcp.py
  MCP setup and MCP outcome policy

nodes/mixed.py
  Oracle retrieval setup and mixed-mode outcome policy
```

## Work items

1. Establish the execution module.
   - Move the shared LLM invocation, tool execution, tool-round limit, OCI message normalization, loop routing, and subgraph builder from `nodes/mixed.py`.
   - Update `chat_agent.py` and both mode modules to import the shared execution module.
   - Preserve graph node names: `mcp_agent` and `mixed_agent`.

2. Introduce the transcript normalization.
   - Add the typed `ToolExecutionTranscript` representation described in `CONTEXT.md`.
   - Centralize pairing tool calls with tool results, identifying tools used, selecting the terminal agent answer, and recording tool failures.
   - Preserve the exact raw AI and tool messages returned by the tool loop; the transcript is derived after the loop and must not replace them.

3. Consume the transcript in outcome policy.
   - Change `nodes/mcp.py` to consume the transcript for tool-use and failure information.
   - Change `nodes/mixed.py` to consume the transcript before applying mixed-only retrieval evidence, reranking, synthesis, citations, and workflow policy.
   - Keep `messages_from_result` as the adapter that produces the existing final message/references contract.

4. Remove obsolete shared implementation from `nodes/mixed.py`.
   - Leave only mixed-mode setup and mixed-mode outcome policy there.
   - Verify that MCP no longer imports generic execution implementation from `nodes/mixed.py`.

5. Update tests.
   - Add focused tests for transcript pairing, terminal-answer selection, tool-failure detection, and missing tool results.
   - Preserve and adjust MCP/mixed node tests so they exercise outcome policy through the transcript.
   - Run the existing workflow coverage for all chat modes to prove graph node wiring and output behavior remain stable.

6. Update project records.
   - Add the completed implementation summary to `CHANGELOG.md` under the current date.
   - Refresh OpenWiki before an agent-created commit, if a commit is requested.

## Acceptance criteria

- Live frontend tool cards still receive one event sequence per tool call, including calls made before the final answer.
- Reopened threads still show completed tool calls from persisted messages.
- MCP and mixed modes retain their current final-answer, references, citations, context-usage, and MCP-invocation result fields.
- The generic execution implementation has one module and is used by both MCP and mixed mode.
- The transcript has direct unit coverage for normal, error, and incomplete tool sequences.
- Relevant unit and workflow tests pass.

## Explicitly out of scope

- Changing the Agent Server SSE/event contract or `@langchain/react` frontend projections.
- Changing the persisted LangGraph state or thread-history schema.
- Replacing the `runtime.context` handoff with private graph state (phase 2).
- Changing Oracle retrieval policy, reranking, citations, or synthesis behavior beyond consuming normalized transcript facts.

## Phase 2 (deferred)

After phase 1 is stable, replace the mutable runtime-context `ToolAgentTurn` convention with a typed private graph-state handoff. Treat it as a separate behavior-preserving change with its own plan and validation.

## Validation completed

- Current-turn transcript scoping regression coverage: a prior turn's tool calls and terminal answer do not affect the current MCP composition.
- `uv run ruff check src/rag_agent/graphs/tool_agent_execution.py src/rag_agent/graphs/nodes/mcp.py src/rag_agent/graphs/nodes/mixed.py src/rag_agent/graphs/chat_agent.py tests/unit_tests/test_tool_agent_turn.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py`
- `uv run pytest tests/unit_tests/test_tool_agent_turn.py tests/unit_tests/test_langgraph_mcp_mixed_nodes.py tests/workflow_tests/test_langgraph_chat_agent_modes.py -q` — 24 passed.

The earlier phase-one validation pass deferred live Agent Server/frontend
verification. That gap was closed by the 2026-08-20 durable recipe-store
validation recorded below and in the recipe-store evidence document.

## Live validation completed

- Started a local source-backed Agent Server on port 2025 with `langgraph dev --allow-blocking`. The flag is development-only and was required because MCP setup performs a synchronous socket connection that LangGraph dev otherwise rejects.
- Ran the real configured calculator MCP path twice. The integration test passed, and a traced `19 + 23` request returned `42` with `mcp_used=true`.
- A streamed `7 + 8` request emitted three `messages/complete` events and persisted `human → ai(tool call) → tool → ai`; the final assistant message retained `calculator_basic_arithmetic` in `mcp_tools_used` and returned `15`.
- Langfuse recorded the complete `chat.request` hierarchy: `mcp_setup → mcp_agent → call_llm → calculator_basic_arithmetic → call_llm → mcp_compose`.

The final focused validation comprised 106 passed tests, plus 2 live Agent
Server MCP/mixed tests. MCP SSE returned `42`; mixed SSE completed retrieval
and calculator execution with 2 citations. Static `mcp_compose` interruption
resumed, and a local process restart against the same checkpoint database
resumed composition with a stable terminal ID and cleared lease. The local
ephemeral thread catalog entry was recreated with the same ID; this is a
local-dev caveat, not production restart evidence. The Langfuse trace was
`aced01a5a7d6770577b0378503fd632f`, and the four-marker IO scan was false for
all markers.

## P2 follow-up: incomplete tool calls

- Transcript normalization now retains a tool call whose matching `ToolMessage` is absent and records it as an explicit incomplete-execution failure.
- The transcript preserves the original AI tool-call order even when a later call completes while an earlier call remains incomplete.
- MCP outcome policy therefore returns the established tool-failure response instead of silently treating the call as unused.
- Regression coverage exercises both the transcript fact and the MCP composition result; targeted lint and unit/workflow validation passed (25 tests).

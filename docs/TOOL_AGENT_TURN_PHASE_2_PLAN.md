# Tool-Agent Turn Phase 2 Plan

## Status

Superseded on 2026-08-20 by [ADR-0001](adr/ADR-0001-TOOL-AGENT-TURN-DURABLE-HANDOFF.md). No implementation or commit has been created.

## Goal

Replace the mutable `runtime.context["tool_agent_turn"]` convention with a typed, ephemeral graph handoff for `ToolAgentTurn` across MCP and mixed-mode execution.

## Why this phase exists

The current handoff has one meaningful seam but an untyped interface:

```text
MCP or mixed setup node
  prepares ToolAgentTurn
  mutates runtime.context["tool_agent_turn"]
      ↓
tool-agent subgraph and compose node
  retrieve and cast it indirectly
```

That makes lifetime, availability, and validation implicit. A missing or incorrectly shaped value fails only when a downstream node reads it. The typed handoff should make the `ToolAgentTurn` interface the test surface and keep setup, execution, and composition local to the same execution path.

## Non-negotiable constraints

- Preserve browser-visible tool-progress events and their ordering.
- Preserve raw AI and tool messages, persisted thread history, graph node names, and result/reference contracts.
- Keep `ToolExecutionTranscript` transient and post-execution.
- Do not put live tool instances, callbacks, or `OracleRetrievalEvidenceStore` into persisted `ChatGraphState` or thread checkpoints.
- Preserve separate MCP and mixed-mode outcome policies.
- Do not alter retrieval, reranking, citation, or synthesis policy as part of this phase.

## Design decision to prove first

`ToolAgentTurn` contains non-serializable, request-scoped values: live tool objects, a runnable configuration, and (in mixed mode) mutable retrieval evidence. Therefore, a field added casually to the parent chat state would risk checkpoint persistence and replay failures.

The selected recovery rule is **durable reconstruction**: persist only a serializable recipe for the request, then rebuild a fresh typed `ToolAgentTurn` when execution resumes. The recipe may contain stable request facts such as mode, selected model, configured MCP server keys, collection selection, and the request message boundary. It must never contain tools, callbacks, runnables, or retrieval evidence.

Before migration, run a narrow LangGraph state-propagation spike that proves one typed private handoff can reach setup, the shared tool-loop subgraph, and composition while remaining absent from persisted thread state and every Agent Server stream payload. The implementation must select the first approach that satisfies all of these conditions:

1. The handoff is typed at its producer and consumers.
2. It is scoped to a single graph invocation.
3. The live `ToolAgentTurn` is not serialized into a checkpointer or replayed as chat state.
4. Neither the live handoff nor its internal representation appears in Agent Server `values`, `updates`, or debug stream payloads.
5. A checkpoint resume rebuilds a fresh handoff before the tool subgraph or compose node consumes it.
6. Concurrent MCP and mixed invocations cannot overwrite one another.
7. The subgraph receives it without reopening a mutable runtime-context convention.

`UntrackedValue` alone does not meet these conditions: it avoids checkpoint persistence but is still ordinary graph state for stream emission and is absent on mid-run resume. If LangGraph private channels cannot satisfy all conditions in this graph shape, stop after the spike and record the constraint before choosing another execution-local adapter. Do not disguise a new mutable context dictionary as graph state.

## Target ownership

```text
tool_agent_turn.py
  constructs and validates the typed ToolAgentTurn

execution-local typed handoff
  owns one invocation's live ToolAgentTurn lifetime and access

serializable turn recipe
  owns only the request facts required to reconstruct a fresh live handoff

tool_agent_execution.py
  consumes the typed handoff inside the shared LLM/tool loop

nodes/mcp.py and nodes/mixed.py
  produce the handoff in setup and consume it in outcome composition
```

## Work items

1. Prove the state model.
   - Build a focused test/spike using the real parent graph and tool-loop subgraph shape.
   - Verify the handoff is available to setup, execution, and compose nodes.
   - Verify a checkpointed run does not persist or replay the live handoff.
   - Inspect Agent Server `values`, `updates`, and debug stream payloads to prove neither the live handoff nor the recipe leaks into the frontend contract.
   - Interrupt after setup and inside the tool subgraph, then resume from the same checkpoint. Verify a fresh handoff is reconstructed before downstream execution.

2. Define the serializable recipe and typed handoff module.
   - Define the minimal serializable recipe required to rebuild one `ToolAgentTurn`; do not duplicate derived values that can be recomputed from persisted messages and runtime request context.
   - Keep `ToolAgentTurn` as the sole representation of prepared execution input.
   - Centralize creation, required-field validation, and missing-handoff errors.
   - Remove the `dict[str, object]` declaration and cast-based runtime lookup from `ChatGraphContext`.

3. Move MCP execution.
   - Make `run_mcp_setup` create the serializable recipe and obtain the live typed handoff through the proven mechanism.
   - Make the MCP tool loop and `run_mcp_compose` consume that handoff directly.

4. Move mixed execution.
   - Apply the identical handoff lifetime to mixed setup, tool loop, and composition.
   - Recreate a fresh `OracleRetrievalEvidenceStore` during reconstruction and preserve the same live instance across retrieval-tool execution and mixed composition within that invocation, without persisting it.

5. Remove the old convention completely.
   - Delete `runtime.context["tool_agent_turn"]` writes.
   - Delete `get_tool_agent_turn(runtime)` and all dependent casts.
   - Confirm no graph module refers to `tool_agent_turn` through runtime context.

6. Validate contracts.
   - Add direct tests for missing/invalid handoff failure and typed lifetime.
   - Update MCP and mixed workflow tests to assert the handoff rather than inspecting mutable runtime context.
   - Run targeted lint, unit, and workflow suites.
   - Run a checkpointed replay regression for both modes.
   - Run real MCP and mixed invocations against the local Agent Server, inspect streamed tool events and persisted messages, and verify new Langfuse traces.

## Acceptance criteria

- No runtime-context key named `tool_agent_turn` remains.
- The handoff is typed end-to-end and has one clear lifetime owner.
- `ToolAgentTurn` never appears in persisted graph state, a reopened thread, or Agent Server `values`, `updates`, or debug stream payloads.
- A serializable recipe contains only stable reconstruction inputs, never live tools, callbacks, runnables, or retrieval evidence.
- MCP and mixed mode retain their existing stream, replay, final answer, citations, retrieval evidence, and MCP invocation contracts.
- Checkpointer-backed tests prove both a fresh request and an interrupted/resumed request rebuild their own live handoff rather than reusing stale request-scoped values.
- Targeted automated checks and live MCP/mixed validation pass.

## Explicitly out of scope

- Replacing the existing raw-message persistence model.
- Changing frontend tool-card projection or Agent Server event types.
- Refactoring `ToolExecutionTranscript` or mode-specific outcome policy.
- Adding new MCP tools, retrieval policy, retries, or approval flows.

## Rollback condition

If no mechanism can both exclude the live handoff from streams and reconstruct it safely after a checkpoint resume, retain the current runtime-context handoff temporarily and document the framework limitation. Do not ship an implementation that serializes live tools or retrieval evidence, leaks them into SSE, or silently loses durable-resume behavior.

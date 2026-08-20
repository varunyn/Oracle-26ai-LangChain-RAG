# ADR-0001: Durable Tool-Agent Turn Handoff

Status: accepted with feasibility gates
Date: 2026-08-20

## Context

MCP and mixed-mode execution need a typed handoff from setup through a shared
tool-loop subgraph to composition. The previous mutable
`runtime.context["tool_agent_turn"]` handoff was process-local and could not
survive worker retry, checkpoint resume, or replay reliably.

Persisting `ToolAgentTurn` in graph state is not acceptable. It contains live
tools and runtime services, while mixed mode also depends on retrieval evidence
that must remain within the established message-artifact contract. Graph state
and checkpoints are user-visible persistence boundaries and feed streaming and
replay behavior.

LangGraph checkpoints at super-step boundaries and may re-execute a node from
its beginning after failure or resume. Agent Server run and runnable IDs are
also not guaranteed to remain the same across every retry or resume
invocation. The design therefore needs stable turn identity, idempotent setup,
fenced concurrency, and cleanup ordered after durable checkpointing.

## Decision

Use a dedicated recipe store owned by `LocalAsyncSqliteSaver` in the existing
Agent Server SQLite database.

The store persists an immutable, versioned recipe keyed by
`(thread_id, turn_id)`, where `turn_id` is the stable ID of the persisted
`HumanMessage` that starts the turn. Agent Server run IDs are stored only as
origin and continuation correlation links.

The recipe contains serializable selection and correlation data only. Each
graph node reconstructs the live `ToolAgentTurn` from the recipe and current
runtime services. No recipe, lease, or live turn enters graph state,
checkpoints, task results, or SSE.

Recipe creation is create-if-absent and idempotent only for an identical
canonical payload. A different payload for an existing key is a conflict.
Selected MCP definitions carry a non-secret version or digest so resume cannot
silently adopt incompatible configuration.

Concurrency uses a renewable five-minute lease with a unique owner and
monotonic fencing token. Renewal uses database time. Every release, terminal
mark, and other lease-sensitive mutation is conditional on the current token.
Long-running nodes renew before half the lease interval and participate in the
Agent Server heartbeat protocol.

The lease prevents concurrent owners where possible but does not promise
exactly-once external tool effects. Tool execution remains at-least-once under
worker loss, and stable tool-call IDs should be used as downstream idempotency
keys when supported.

Composition marks a stable terminal message ID but never deletes the recipe
directly. The pinned SQLite saver cannot prove generic post-checkpoint
quiescence because `aput_writes()` may commit later and pull-task scheduling
requires graph topology unavailable to the saver. It therefore records
checkpoint-to-turn reachability atomically with checkpoint insertion and keeps
terminal recipes until all checkpoint and run links are gone. Thread deletion,
run rollback, pruning, copying, and orphan cleanup are implemented under the
saver's existing SQLite lock and transaction boundary.
The saver performs conservative cleanup at startup and periodically while it
is alive; database transactions serialize multiple Agent Server workers.

The complete contract is defined in
`docs/TOOL_AGENT_TURN_RECIPE_STORE_SPEC.md`.

## Required Feasibility Gates

Implementation must stop if any of these cannot be demonstrated against the
installed LangGraph Agent Server and checkpointer:

1. A stable persisted user-message ID is available unchanged in setup, the
   parent graph, the shared subgraph, retry, interrupt/resume, replay, and
   composition.
2. Fresh per-claim random owner IDs and persisted fences reject stale worker
   and node attempts without relying on a runnable `run_id`.
3. Long-running node work can renew the lease and runtime heartbeat safely.
4. Terminal checkpoint durability can be observed directly or established
   conservatively by a reconciler.
5. Checkpoint and recipe lifecycle mutations can share one saver transaction
   without nested-lock deadlocks.

If the first gate fails, introduce a stable private execution identity at the
Agent Server boundary. Do not use graph state, mutable runtime context, or
runnable `run_id` as a fallback.

## Consequences

### Positive

- Worker retries and resumed runs recover original selections.
- Live runtime objects remain outside persistence and streaming contracts.
- Idempotent creation and fenced leases make replay and takeover explicit.
- Terminal cleanup cannot make a pre-terminal checkpoint unrecoverable.
- Recipe lifecycle follows checkpoint lifecycle through one owner and
  transaction boundary.

### Costs

- The custom saver owns an additional schema, reconciliation process, and
  lifecycle tests.
- MCP configuration needs stable non-secret versioning.
- Long-running work needs lease renewal, heartbeat, and cancellation handling.
- Tool effects cannot be advertised as exactly once.
- Copy, prune, rollback, and retention semantics require Agent Server
  integration tests rather than unit tests alone.

## Rejected Alternatives

### Persist `ToolAgentTurn` in graph state

Rejected because it contains live, non-serializable runtime objects and would
change checkpoint, replay, and SSE contracts.

### Keep mutable `runtime.context`

Rejected because it is process-local and not a durable retry/resume handoff.

### Key only by Agent Server or runnable run ID

Rejected because retry and resume invocations can receive different IDs and
would be unable to recover the original recipe reliably.

### Delete from composition

Rejected because composition returns before its terminal checkpoint is
durably committed. A crash in that window would leave a resumable checkpoint
without its recipe.

### Use an unfenced fixed-duration lease

Rejected because slow tools can outlive the lease, and a stale worker could
then release or finalize work owned by its successor.

### Store recipes in a separate SQLite connection

Rejected because checkpoint and recipe lifecycle mutations could interleave or
commit independently, producing orphaned or prematurely deleted rows.

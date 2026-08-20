# Tool-Agent Turn Feasibility Evidence

Date: 2026-08-20
Probe: `tests/workflow_tests/test_tool_agent_turn_feasibility.py`
Runtime: LangGraph 1.2.6, langgraph-checkpoint-sqlite 3.1.0, LangChain Core 1.4.8.

## Results

The explicit-ID identity probe passes. A `HumanMessage(id="turn-1")` remains
`turn-1` in setup, the parent-to-subgraph boundary, the subgraph node, the
interrupt/resume execution, the replayed checkpoint, and composition. A second
user message with `id="turn-2"` is distinct. The persisted replay contains the
original ID, and the subgraph execution reports its checkpoint namespace
separately where LangGraph creates one.

The checkpointer probe also passes for an id-less input: LangGraph assigns a
message ID before setup, and that generated ID is then visible in every node.
The stateless graph also receives an ephemeral generated ID, but has no durable
checkpoint in which to retain it. Therefore the identity gate is satisfied for
checkpointed Agent Server requests; stateless tool-agent requests still need
an API-level durable identity or must be rejected before setup. The generated-ID
behavior should be covered by the Agent Server probe before the recipe store is
started.

The execution-info probe passes for the locally available identity material.
Each node receives `thread_id`, `checkpoint_ns`, `task_id`, and `node_attempt`;
these fields can distinguish node attempts without using only `run_id`. A
synthetic owner token built from those fields does not occur in debug stream
payloads. The probe cannot prove an Agent Server worker/resource identity:
the open-source graph execution supplies no server worker field, so a lease
owner contract still needs an Agent Server-level attempt/resource identifier.

Terminal durability is observable after graph completion. The latest saver
checkpoint contains the stable terminal `AIMessage.id`, has checkpoint
metadata, and has no pending writes. This supports a saver-owned reconciler
that checks the terminal message and pending work after checkpoint commit; it
does not justify synchronous deletion from composition.

The transaction probe passes when lifecycle SQL executes under the saver’s
existing lock and connection: a transaction rollback removes both a lifecycle
row and checkpoint row. `AsyncSqliteSaver.lock` is a non-reentrant
`asyncio.Lock`, so store methods called while the saver lock is held must use
internal SQL helpers and must not call public saver methods that reacquire the
lock. This is feasible, but nested-lock behavior must remain an explicit
implementation constraint.

The two-connection lease probes pass. Simultaneous `BEGIN IMMEDIATE` claim
attempts against one recipe row produce exactly one winner, even though each
attempt has a fresh random owner ID. The winning row has fence `1`. After
forcing expiry, a second owner takes over at fence `2`; a mutation using the
old owner and fence affects zero rows, and the successor’s owner, fence, and
expiry remain unchanged. This proves the SQLite compare-and-swap shape needed
for claim and stale-token fencing without implementing the recipe store.

## Gate decision

Ticket 01 is now unblocked for Ticket 02. The durable checkpoint, transaction,
fresh-owner, and monotonic-fence seams are feasible. Ticket 02 can begin the
saver-owned recipe-store foundation, retaining the tested constraints that
claim attempts use fresh random owner IDs and all lease-sensitive mutations
compare both owner and fence.

The remaining stateless-message policy is an API contract to preserve during
Ticket 02; it is not a lease-owner blocker. The probes are intentionally
limited to feasibility and do not add a recipe schema, store interface, lease
implementation, or finalizer.

## Durable recipe-store validation

The implemented Tickets 02–05 were validated on 2026-08-20 with 106 focused
tests and five live Agent Server MCP/mixed tests. MCP SSE returned `42`; mixed
SSE completed retrieval plus calculator execution and returned 2 citations.
Neither stream or its persisted state exposed `tool_agent_turn`,
`lease_owner_id`, `recipe_json`, or `client_secret` markers. A static interrupt
at `mcp_compose` resumed successfully.

For local restart recovery, the process was stopped after the tool loop and
restarted against the same `.local-data/langgraph-checkpoints.sqlite`.
Because the local-dev thread catalog is ephemeral, its missing entry was
recreated with the same ID before composition resumed to `42`. The terminal
message ID stayed stable, and the SQLite recipe contained that
`terminal_message_id`, had no lease owner, and retained checkpoint links.
This is local-dev catalog recreation evidence only; it is not a production
restart or production catalog-recovery claim.

Langfuse trace `aced01a5a7d6770577b0378503fd632f` showed
`chat_agent -> mcp_setup -> mcp_agent -> call_llm ->
calculator_basic_arithmetic -> call_llm -> mcp_compose`; the IO-marker scan
was false for all four markers.

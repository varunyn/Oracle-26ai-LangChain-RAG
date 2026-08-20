# Tool-Agent Turn Durable Recipe Store Specification

Status: implemented, revised 2026-08-20. Tickets 02–05 implement durable
reconstruction, leases, replay fidelity, and conservative checkpoint-aware
lifecycle cleanup. Immediate terminal finalization remains unavailable in the
pinned SQLite saver topology and is intentionally deferred.

## Ticket 04 Checklist

Status: implemented on 2026-08-20.

- [x] Preserve checkpointed Oracle retrieval failures as an explicit error
  discriminator rather than conflating them with empty context.
- [x] Persist implicit MCP selection as the configured server-key set and
  reject deleted or changed definitions during reconstruction.
- [x] Keep setup idempotent and conflict-detecting while making load-only
  reconstruction independent of mutable defaults.
- [x] Preserve additive `enable_reranker` recipe compatibility for older rows.

Deterministic evidence is recorded in the focused replay and reconstruction
tests: retrieval failure versus empty context, implicit-selection deletion
drift, load-only reconstruction with unavailable mutable defaults, and
backward-compatible reranker fields.

## Ticket 05 Checklist

- Status: implemented on 2026-08-20.

- [x] Atomically link each recipe-relevant checkpoint to its durable turn.
- [x] Keep terminal recipes while a retained checkpoint or run can reach them.
- [x] Apply thread deletion, rollback, pruning, copying, and reconciliation
  through the saver transaction seam.
- [x] Reject copying an active recipe and reset copied lifecycle fields.
- [x] Document the unavailable immediate-finalization gate.
- [ ] Revisit immediate finalization only if LangGraph exposes a durable
  completed-superstep signal to the custom saver.

Validation completed on 2026-08-20:

- 106 focused tests passed; the live Agent Server MCP and mixed-mode tests
  passed (2 tests).
- MCP SSE completed with `42`. Mixed SSE completed with retrieval plus the
  calculator tool and returned 2 citations. The stream, payload, and stored
  state scans found no `tool_agent_turn`, `lease_owner_id`, `recipe_json`, or
  `client_secret` markers.
- A static interrupt at `mcp_compose` resumed successfully.
- A local process was stopped after the tool loop and restarted against the
  same `.local-data/langgraph-checkpoints.sqlite`. In the local-dev ephemeral
  thread catalog, the missing entry was recreated with the same thread ID;
  this catalog recreation is a local-dev recovery detail, not evidence of a
  production restart guarantee. Composition then resumed to `42` with the
  stable terminal message ID; the SQLite recipe had that
  `terminal_message_id`, a cleared lease, and retained checkpoint links.
- Langfuse trace `aced01a5a7d6770577b0378503fd632f` retained the hierarchy
  `chat_agent -> mcp_setup -> mcp_agent -> call_llm ->
  calculator_basic_arithmetic -> call_llm -> mcp_compose`. The IO-marker scan
  was false for all four markers.

## Purpose

Replace the mutable `runtime.context["tool_agent_turn"]` handoff with a durable,
typed recipe that setup, the shared tool-loop subgraph, and composition can use
across worker retries and resumptions.

The recipe stores only serializable selection and correlation data. The live
`ToolAgentTurn` remains ephemeral and is reconstructed inside each node.

## Non-Negotiable Invariants

1. Neither the recipe, the live `ToolAgentTurn`, nor lease data may enter
   `ChatGraphState`, LangGraph checkpoints, task results, or SSE events.
2. A resumed turn uses its original selections. It must not silently adopt
   changed request context or MCP configuration.
3. Recipe creation is idempotent for the same canonical payload and fails on a
   conflicting payload.
4. Every mutation after a lease claim is fenced. A stale worker cannot renew,
   release, finalize, or delete a recipe owned by a successor.
5. A recipe is not deleted until the terminal graph result is durably
   checkpointed.
6. Tool execution is at-least-once. The store coordinates workers but cannot
   promise exactly-once effects across process loss or an external tool
   boundary.
7. Recipe lifecycle operations share the checkpointer's SQLite transaction and
   lock. Checkpoint and recipe cleanup cannot be independently committed.

## Ownership and Public Seam

`LocalAsyncSqliteSaver` owns the recipe-store schema, SQLite connection,
transaction boundary, and lock. The graph depends only on a typed
`ToolAgentTurnRecipeStore` collaborator exposed by the saver/runtime
integration.

The minimum interface is:

```python
create_or_load(recipe) -> Created | ExistingIdentical | RecipeConflict
load(key) -> ToolAgentTurnRecipe | Missing
claim(key, owner_id) -> ClaimedRecipe | AlreadyActive | Missing
renew(key, lease_token) -> Renewed | StaleLease | Missing
release(key, lease_token) -> Released | StaleLease | Missing
mark_terminal(key, lease_token, terminal_message_id) -> Marked | StaleLease
delete_for_origin_runs(run_ids) -> None
delete_for_thread(thread_id) -> None
reconcile() -> ReconciliationResult
```

`claim` returns an opaque lease token containing the owner and monotonically
increasing fence. All later lease-sensitive operations use that exact token.

## Durable Identity

The recipe key is `(thread_id, turn_id)`.

- `thread_id` is the LangGraph thread identifier.
- `turn_id` is the stable ID of the latest persisted `HumanMessage` that
  started the tool-agent turn.
- Agent Server run IDs are correlation and cleanup links, not the sole recipe
  identity.

This avoids depending on a runnable `run_id`, which may change across worker
attempts or a resumed Agent Server invocation.

### Identity Feasibility Gate

Before implementation, an integration probe must prove that the chosen
`HumanMessage.id`:

1. is present and identical in setup, the parent graph, every tool-loop
   subgraph node, and composition;
2. survives checkpoint reload, worker retry, interrupt/resume, and thread
   replay;
3. changes for a new user turn;
4. is unaffected by checkpoint namespaces; and
5. is available before recipe creation.

Stateless calls without a durable `thread_id` and stable user-message ID must
either receive an explicitly generated durable turn identity at the API
boundary or be rejected for tool-agent modes.

If this gate fails, stop the implementation and define a stable private
execution identity at the Agent Server boundary. Do not fall back to
`runtime.context`, runnable `run_id`, or graph state.

Each Agent Server run invocation associated with a turn is recorded in a
separate run-link row. The first invocation is the origin run; retries and
resume invocations are continuation links.

## Recipe Contents

Allowed fields are limited to:

- schema version;
- thread ID, turn ID, origin run ID, and request/session correlation IDs;
- selected chat mode and model key;
- selected collection key;
- selected MCP server keys;
- non-secret MCP configuration version or digest;
- tracing policy, tool-round limit, and `enable_reranker`; and
- database-generated creation timestamp.

`enable_reranker` was added to schema version 1 as an additive field. Rows
written before the field existed deserialize it as `false`, preserving the
historical non-reranked behavior. Resume never substitutes the current
request's value for a missing field.

The recipe must not contain user text, prompts, credentials, server URLs,
headers, callbacks, live tool objects, model clients, retrieved documents,
tool results, or retrieval evidence.

MCP reconstruction resolves empty requested selection to all configured
adapter-server keys at recipe creation and persists those keys plus a
secret-free behavioral compatibility digest. The digest recursively projects
all definition data, including transport options, arguments, environment,
session/client settings, encoding, and tool-selection/definition fields, while
structurally redacting credential-bearing fields and secret header values. It
retains behavioral identity such as `client_id` and non-secret routing,
tenant, or version headers. It therefore excludes bearer tokens, client
secrets, and secret header contents, so routine credential rotation does not
invalidate resume, while behavior-bearing configuration drift remains visible.
It requires every recorded key to exist in the authoritative live definitions
before accepting a digest. If a referenced definition was removed or its
endpoint, transport, or tool-definition behavior changed, reconstruction
fails explicitly instead of using the current definition silently.

## SQLite Schema

The initial schema is logically equivalent to:

```sql
CREATE TABLE tool_agent_turn_recipes (
    thread_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    origin_run_id TEXT,
    recipe_json BLOB NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at_epoch_ms INTEGER NOT NULL,
    lease_owner_id TEXT,
    lease_fence INTEGER NOT NULL DEFAULT 0,
    lease_expires_at_epoch_ms INTEGER,
    terminal_message_id TEXT,
    terminal_marked_at_epoch_ms INTEGER,
    PRIMARY KEY (thread_id, turn_id)
);

CREATE INDEX tool_agent_turn_recipes_origin_run_idx
    ON tool_agent_turn_recipes(origin_run_id);

CREATE TABLE tool_agent_turn_run_links (
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    created_at_epoch_ms INTEGER NOT NULL,
    PRIMARY KEY (thread_id, run_id, turn_id),
    FOREIGN KEY (thread_id, turn_id)
        REFERENCES tool_agent_turn_recipes(thread_id, turn_id)
        ON DELETE CASCADE
);

CREATE INDEX tool_agent_turn_run_links_run_idx
    ON tool_agent_turn_run_links(run_id);
```

Use database UTC time expressed as integer epoch milliseconds for creation and
lease comparisons. Do not compare worker-supplied wall clocks or free-form
timestamp text.

`create_or_load` inserts if absent. On an existing key it compares a
canonical, versioned serialization:

- an identical payload returns `ExistingIdentical`;
- a different payload returns `RecipeConflict` and does not overwrite.

Claim, renewal, release, and terminal marking are compare-and-swap
transactions. An expired-lease takeover increments `lease_fence`.

## Lease and Concurrency Contract

The renewable lease duration is five minutes.

Each claim attempt generates a fresh random owner ID. It is intentionally
ephemeral: durability belongs to the persisted monotonic fence, not the owner
string. The owner ID must be unique for a worker/node attempt and must not
rely only on an unstable runnable `run_id`.

While a node is active:

1. claim the recipe before reconstructing live dependencies;
2. renew before half the lease interval elapses and call the runtime heartbeat
   mechanism while long LLM or tool work is in progress;
3. verify the lease fence immediately before each external tool side effect
   and before terminal marking; and
4. release conditionally on known interrupt, cancellation, or ordinary
   nonterminal completion.

If a worker disappears, lease expiry permits takeover. The old token becomes
invalid permanently. A stale worker must fail closed when it next renews or
attempts a lease-sensitive mutation.

This protocol reduces concurrent duplicate work but does not create
exactly-once external effects. Tool calls should carry their stable
`tool_call_id` as an idempotency key where the tool supports one.

## Graph Integration

### Setup

Setup derives the durable turn key, constructs the canonical recipe, and calls
`create_or_load`. Replayed setup accepts only an identical recipe. It also
records the current Agent Server run as an origin or continuation link.

### Tool Loop

Every shared tool-loop node derives the same turn key, claims the recipe, and
reconstructs `ToolAgentTurn` from the recipe plus current runtime services.
Original recipe selections override mutable request defaults during resume.

Node output contains only the existing graph-state updates. Recipe and lease
objects remain runtime-local.

LangGraph checkpoints at super-step boundaries and may rerun a node from its
beginning after failure or resume. Setup and tool-loop side effects must
therefore be idempotent under re-execution.

### Mixed-Mode Evidence

Mixed composition reconstructs Oracle retrieval evidence from the current
turn's persisted `oracle_retrieval` `ToolMessage` artifact. Recipe storage
does not duplicate retrieved documents or evidence. Retrieval failures use a
serializable `oracle_retrieval_error` artifact discriminator so replay keeps
the retrieval-failure answer distinct from a successful empty result.

### Composition and Durable Finalization

Composition claims the recipe and returns a final `AIMessage` with a stable
message ID. It conditionally records that ID with `mark_terminal`, but it
does not delete the recipe.

A saver-owned checkpoint-link index records which retained checkpoints can
reconstruct each recipe. Terminal recipes are retained while any such link
exists. Lifecycle deletion is allowed only after every checkpoint link and run
link is gone and no live lease exists. A crash before or after a terminal
checkpoint therefore leaves a recoverable recipe until checkpoint lifecycle
cleanup removes its reachability.

### Finalization Feasibility Gate

The installed LangGraph 1.2.6 / SQLite saver 3.1.0 topology does not expose a
durable post-checkpoint quiescence signal. `aput()` commits independently from
later `aput_writes()` calls, and the saver lacks graph topology needed to infer
pull-task scheduling from persisted channel versions. The approved fallback is
checkpoint-link retention: never delete synchronously from composition or
`aput()`, and reconcile only unlinked inactive rows after grace/retention.

## Checkpointer Lifecycle

All lifecycle behavior is implemented in the saver under its existing
connection and lock, with one SQLite transaction per checkpoint/recipe
operation. A pinned local `aput()` copy keeps checkpoint insertion and
checkpoint-link insertion in that same transaction without nested saver locks.

- `adelete_thread`: delete checkpoints, writes, run links, and every recipe
  for the thread atomically.
- `adelete_for_runs`: deleting an origin run may delete its recipe only when
  no surviving checkpoint can resume the turn. Deleting a continuation run
  removes its link, not the shared recipe.
- `aprune(delete)`: behave like thread deletion for recipes.
- `aprune(keep_latest)`: preserve recipes required by every retained checkpoint
  because generic saver persistence cannot prove terminal quiescence.
- `acopy_thread`: copy every recipe required by a copied nonterminal
  checkpoint under the target thread, clear lease and terminal fields, and
  record source provenance. If the saver cannot prove a consistent copy, reject
  copying an active tool-agent turn.
- failed-run rollback: remove run links and delete an orphaned recipe only when
  no checkpoint or other run link can resume it.

The reconciler removes:

- recipes created before a setup checkpoint that never became reachable;
- unlinked terminal recipes after the terminal retention period;
- expired run links and recipes with no reachable checkpoint or live link; and
- recipes whose thread was removed outside the normal lifecycle hook.

Retention must be at least as long as the maximum checkpoint/thread retention
window. The saver invokes reconciliation once at startup and from one bounded
periodic task per saver; SQLite `BEGIN IMMEDIATE` serializes concurrent server
connections. Shutdown cancels and awaits that task. Reconciliation logs
identifiers and reason codes, never recipe payloads.

## Failure Behavior

- Missing recipe for a resumable tool-agent checkpoint: fail explicitly and
  preserve the checkpoint for diagnosis.
- Conflicting setup replay: fail with `RecipeConflict`.
- Active lease owned elsewhere: wait/retry according to Agent Server retry
  policy; do not run the node concurrently.
- Lease lost during work: cancel when possible and reject all later mutations.
- Missing or incompatible MCP configuration version: fail explicitly.
- Unsupported recipe schema: fail with a migration/version error.
- Store unavailable or locked beyond retry policy: fail the node; do not fall
  back to mutable runtime context.

## Validation Requirements

### Identity and Reconstruction

- The same turn key is observed in parent and subgraph nodes, worker retry,
  interrupt/resume, replay, and composition.
- A new user turn receives a distinct key.
- Stateless behavior is explicitly supported or rejected.
- Reconstruction uses the original model, collection, MCP selections, and
  configuration version after request defaults change.
- Changed or deleted MCP definitions fail explicitly.

### Creation and Crash Recovery

- Replayed identical setup is idempotent.
- Replayed conflicting setup fails without overwriting.
- A crash after recipe creation but before setup checkpoint is reconciled.
- A crash after composition returns but before checkpoint commit can resume.
- A crash after terminal checkpoint retains the recipe while its checkpoint
  link exists; cleanup occurs only after reachability is removed.

### Lease and Concurrency

- Two simultaneous claims produce one owner.
- Renewal keeps a tool running longer than five minutes from being taken over.
- Expired takeover increments the fence.
- A stale owner cannot renew, release, mark terminal, or delete.
- Cancellation and interrupt release paths are exercised.
- A worker loss during an external tool call demonstrates and documents
  at-least-once behavior.

### Checkpointer Lifecycle

- Thread deletion, checkpoint-link cleanup, and recipe cleanup are one
  transaction.
- Origin-run rollback, continuation-run deletion, prune-delete,
  prune-keep-latest, and orphan cleanup preserve only resumable recipes.
- Copying a nonterminal thread copies a consistent recipe or is rejected.
- Boundary lease expiry uses database time and treats equality consistently.
- Nested store/checkpointer calls cannot deadlock the saver lock.

### State and Streaming

- Checkpoint payloads contain no recipe, live turn, lease, credentials, or
  retrieved evidence beyond the established `ToolMessage` artifact contract.
- SSE events and task results expose none of those objects.
- Stream shape, visible tool progress, final answers, citations, and replay
  remain unchanged.

### End-to-End

- MCP and mixed requests complete through Agent Server with real tool calls.
- A process restart between setup, tool-loop nodes, and composition resumes
  from the original recipe.
- Live Langfuse hierarchy remains
  `chat_agent -> setup -> tool loop -> composition` without recipe payloads
  in observation input/output.

The 2026-08-20 live validation above exercised these checks on the local
Agent Server and local SQLite saver. It does not claim a production process
restart or production thread-catalog recovery guarantee.

## Out of Scope

- Legacy runtime-context fallback or migration.
- Exactly-once guarantees for external tools.
- Persisting live tools, clients, callbacks, credentials, or retrieval evidence.
- Historical time travel into a terminal tool-agent turn after its recipe has
  been safely finalized.
- Changes to public graph-state or SSE schemas.

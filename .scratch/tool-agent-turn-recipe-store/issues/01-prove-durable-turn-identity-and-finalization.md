# 01 — Prove durable turn identity and finalization feasibility

**What to build:** Demonstrate that the active Agent Server/checkpointer topology can supply stable turn identity, safe lease-owner identity, and a post-checkpoint finalization signal for tool-agent runs.

**Blocked by:** None — can start immediately.

**Status:** feasibility-probed; unblocked for Ticket 02

- [x] Prove or reject stable persisted user-message identity across parent graph, subgraph, resume, replay, and a new turn.
- [x] Prove or reject safe lease-owner and terminal-checkpoint finalization signals without exposing internal data to streams.

Evidence is recorded in `docs/TOOL_AGENT_TURN_FEASIBILITY_EVIDENCE.md` and the
workflow probes in `tests/workflow_tests/test_tool_agent_turn_feasibility.py`.
Explicit and checkpointer-generated message IDs plus post-checkpoint inspection
pass. Stateless generated IDs remain ephemeral and must be rejected or made
durable at the API boundary. The lease-owner decision is fresh random owner
IDs plus the persisted monotonic fence.

Follow-up probes now use two independent SQLite connections. Simultaneous
claims produce one winner, expired takeover increments the fence, and every
stale owner/fence mutation is rejected. The lease-owner decision is fresh
random owner IDs plus the persisted monotonic fence. Ticket 01 is unblocked
for Ticket 02; preserve the stateless-message policy as an API contract.

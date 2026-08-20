# 02 — Add saver-owned recipe-store foundation

**What to build:** Give the existing SQLite checkpointer an atomic, durable recipe-store implementation with immutable recipe creation and fenced renewable leases.

**Blocked by:** 01 — Prove durable turn identity and finalization feasibility.

**Status:** implemented; graph integration deferred to Ticket 03

- [x] Persist only the approved serializable recipe and its lifecycle metadata under the checkpointer's transaction and lock.
- [x] Demonstrate idempotent creation, conflict rejection, claim/renew/release, stale-owner fencing, and cleanup primitives.

Implemented in `src/rag_agent/runtime/tool_agent_recipe_store.py` and exposed
through `LocalAsyncSqliteSaver.recipe_store`. Focused coverage is in
`tests/unit_tests/test_tool_agent_recipe_store.py`; existing checkpointer and
feasibility suites remain green. No graph reconstruction or stream contract
changes are included. Recipe transactions now roll back on unexpected failure,
SQLite foreign keys are enabled, continuation run links are idempotent, and
origin-run cleanup preserves recipes that still have resumable continuation
links; orphaned terminal cleanup remains reconciliation scope.

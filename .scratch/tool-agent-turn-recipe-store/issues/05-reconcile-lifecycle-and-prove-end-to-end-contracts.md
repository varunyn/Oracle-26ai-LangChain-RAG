# 05 — Reconcile lifecycle and prove end-to-end contracts

**What to build:** Safely finalize and reconcile durable recipes while proving concurrency, cleanup, streaming, replay, and live MCP/mixed behavior.

**Blocked by:** 02 — Add saver-owned recipe-store foundation; 03 — Reconstruct tool-agent turns through the tool loop; 04 — Preserve mixed evidence and configuration fidelity.

**Status:** implemented — validated

- [x] Cover terminal cleanup, run/thread deletion, prune/copy/rollback, and orphan reconciliation without prematurely deleting resumable recipes.
- [x] Deterministically prove recipe and live-turn data do not leak into graph state, checkpoints, persisted messages, or stream/frontend projections.
- [x] Run live MCP/mixed, SSE/replay, and Langfuse validation. 106 focused tests and 5 live Agent Server MCP/mixed tests passed; MCP SSE returned `42`, mixed retrieval plus calculator returned 2 citations, static `mcp_compose` interruption resumed, and Langfuse trace `aced01a5a7d6770577b0378503fd632f` showed the expected graph/tool hierarchy with no recipe, lease, tool-agent, or secret markers. A local process restart against the same checkpoint database resumed composition with a stable terminal ID and cleared lease; the local-dev ephemeral thread-catalog entry was recreated with the same ID, which is a local-dev caveat rather than a production restart guarantee.

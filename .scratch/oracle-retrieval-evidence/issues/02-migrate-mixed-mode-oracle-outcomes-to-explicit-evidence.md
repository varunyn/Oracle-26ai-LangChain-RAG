# 02 — Migrate mixed-mode Oracle outcomes to explicit evidence

**What to build:** Mixed-mode chat records Oracle lookup evidence through the new interface and consumes it directly to produce the existing successful-answer, citations, reranking, synthesis fallback, no-context, and retrieval-failure outcomes—while preserving MCP results. The private `_retrieval_state` handoff is removed, with unit and compiled-graph workflow coverage for invocation linkage, repeat calls, assistant-message metadata, and replay-safe final citations.

**Blocked by:** 01 — Create the Oracle retrieval-evidence seam.

**Status:** ready-for-agent

- [ ] A successful Oracle lookup in mixed mode produces an answer with citations from the documents belonging to the selected invocation, without scanning loaded tools for private state.
- [ ] An empty Oracle lookup retains the existing no-context outcome; a failed lookup retains the existing retrieval-failure outcome; MCP results remain available to mixed-mode composition.
- [ ] Reranking and synthesis fallback consume selected explicit evidence, tool invocation metadata remains present, and no `_retrieval_state` handoff remains.
- [ ] Deterministic unit and compiled-graph workflow tests cover these outcomes without real Oracle, MCP, or model calls.

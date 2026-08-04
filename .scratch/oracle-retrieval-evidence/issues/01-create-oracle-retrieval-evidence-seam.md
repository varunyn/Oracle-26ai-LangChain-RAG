# 01 — Create the Oracle retrieval-evidence seam

**What to build:** A single typed, turn-scoped interface that records and retrieves the latest completed Oracle lookup, explicitly linked to its tool invocation. It represents documents, empty results, and retrieval errors without exposing tool-private state, with deterministic contract tests.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A caller can record and read the latest completed Oracle retrieval evidence, including its invocation identifier, query, documents, and optional error.
- [ ] An empty documents collection without an error is distinguishable from a retrieval failure, and repeated retrieval calls have a deterministic latest-completed selection rule.
- [ ] The interface contract is verified without inspecting storage or a retrieval tool's private attributes.

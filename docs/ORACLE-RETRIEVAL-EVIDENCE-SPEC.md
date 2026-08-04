# Oracle Retrieval-Evidence Module

## Problem Statement

When a chat user asks a mixed-mode question, the application may need both Oracle collection evidence and MCP tool results. Today, Oracle retrieval keeps its documents and failure details inside private state attached to the retrieval tool. The mixed-mode outcome code later searches loaded tools for that private state before it can decide whether to rerank evidence, produce citations, synthesize an answer, or return a controlled no-context or retrieval-failure response.

This hidden handoff makes Oracle retrieval behavior hard to follow and fragile to change. A user can receive an incorrect missing-context or retrieval-failure result if the retrieval evidence is not recovered consistently. It also prevents the application from clearly connecting a retrieval result to the tool invocation that produced it.

## Solution

Introduce one deep Oracle retrieval-evidence module. The Oracle retrieval tool will record a structured retrieval-evidence value through this module. Mixed-mode outcome construction will consume that value through the same explicit interface instead of inspecting private state on loaded tools.

The value will describe the Oracle collection retrieval result for a turn: its invocation linkage, retrieved documents, query, and error or empty-result state. The module will own evidence collection and retrieval; mixed-mode outcome construction will own only the user-visible decision based on that evidence.

## User Stories

1. As a chat user, I want mixed mode to use Oracle collection evidence reliably, so that my answer reflects the collection result.
2. As a chat user, I want an Oracle retrieval failure to be distinguished from an empty collection result, so that I receive the right controlled response.
3. As a chat user, I want citations to correspond to the Oracle documents used for my answer, so that I can verify the result.
4. As a chat user, I want an empty Oracle retrieval result to produce the existing no-context outcome, so that the application does not invent collection facts.
5. As a chat user, I want mixed mode to preserve useful MCP tool results alongside Oracle evidence, so that a multi-step request can complete coherently.
6. As a chat user, I want reranking to use the documents returned by the relevant Oracle retrieval invocation, so that answer quality remains consistent.
7. As a chat user, I want retries or repeated Oracle retrieval calls to remain traceable, so that the final answer is based on known evidence.
8. As a chat user, I want thread replay to retain the final answer and citations without depending on hidden tool state, so that refreshing chat does not alter what I see.
9. As an application developer, I want one explicit retrieval-evidence interface, so that Oracle retrieval results do not leak through a tool's private implementation.
10. As an application developer, I want one place to represent Oracle documents, errors, and invocation linkage, so that retrieval behavior has high locality.
11. As an application developer, I want mixed-mode composition to consume evidence without scanning arbitrary tools, so that adding MCP tools cannot alter Oracle result discovery.
12. As an application developer, I want retrieval-policy checks to receive structured evidence, so that no-context and failure logic stays deterministic.
13. As an application developer, I want tests to create evidence through the public retrieval-evidence interface, so that they survive retrieval-tool refactors.
14. As an application developer, I want the retrieval tool to remain responsible for executing Oracle collection lookup, so that the new module does not duplicate retrieval implementation.
15. As an operator, I want a retrieval result linked to its tool invocation, so that logs and traces explain which lookup produced an answer or failure.
16. As an operator, I want Oracle collection and query information retained in the evidence value where safe, so that retrieval failures can be diagnosed without reverse-engineering tool internals.
17. As a future maintainer, I want private retrieval-state attributes deleted after migration, so that there is one authoritative evidence path.
18. As a future maintainer, I want multiple Oracle retrieval calls to have an explicit selection rule, so that their evidence cannot be accidentally conflated.

## Implementation Decisions

- Add one deep Oracle retrieval-evidence module with a single typed interface for recording and reading retrieval evidence during a mixed-mode turn.
- The retrieval-evidence value will include an Oracle retrieval invocation identifier, query, retrieved documents, and an optional error. An empty document list without an error represents a completed no-context lookup.
- The Oracle retrieval tool will write evidence through this interface as it executes. It will no longer expose private state for composition code to inspect.
- Mixed-mode outcome construction will read the retrieval-evidence value directly. It will not search the configured tool collection or use attribute inspection to discover Oracle evidence.
- Existing policy evaluation will receive the structured evidence needed to distinguish successful retrieval, no-context retrieval, and retrieval failure.
- Existing reranking, citation normalization, synthesis, final-answer metadata, and user-facing controlled messages will retain their current contracts.
- The module will make evidence selection explicit. The initial rule is to retain the latest completed Oracle retrieval evidence for the turn, linked to the corresponding invocation. The implementation must make a future multiple-invocation policy possible without another hidden handoff.
- Oracle collection lookup remains in the retrieval implementation. MCP tool loading, shared tool-turn preparation, direct mode, and RAG-only mode remain unchanged.
- No database schema, request field, public response field, or frontend rendering contract changes are required.

## Testing Decisions

- The one new test seam is the Oracle retrieval-evidence interface. Tests will record or retrieve evidence through that interface and assert user-visible mixed-mode outcomes.
- Good tests assert externally meaningful behavior: retrieved evidence produces citations and a successful answer; empty evidence produces the controlled no-context response; error evidence produces the controlled retrieval-failure response.
- Tests will not inspect private tool attributes, assert arbitrary-tool scan order, or depend on internal storage details.
- Add deterministic tests linking an Oracle tool invocation to documents, an empty result, and an error result.
- Add mixed-mode outcome tests proving that reranking, synthesis fallback, citations, and policy decisions consume the explicit evidence value.
- Preserve workflow coverage for the compiled chat graph and the existing MCP/mixed mode contract, including tool invocation metadata and final assistant-message references.
- Prior art is the existing mixed-node unit coverage, graph-mode workflow coverage, and citation contract tests. Use fake tools and fake model responses; no real Oracle database, MCP server, or model call is needed for these deterministic tests.

## Out of Scope

- Changing Oracle document retrieval algorithms, collection selection, filter semantics, reranker behavior, or citation normalization.
- Changing MCP server configuration, tool loading, or the shared MCP tool-turn preparation module.
- Changing direct mode, RAG-only mode, frontend stream projection, tool-call rendering, or thread persistence.
- Adding a database table or long-lived persistence for transient retrieval evidence.
- Changing user-facing wording for the existing Oracle no-context and retrieval-failure outcomes.
- Designing the final policy for aggregating multiple successful Oracle retrieval calls beyond making selection explicit and extensible.

## Further Notes

- The agreed highest seam is the Oracle retrieval-evidence interface between the retrieval tool and mixed-mode outcome construction. This is the only new seam proposed by this spec.
- The current private `_retrieval_state` handoff is a shallow seam: deleting it exposes needed behavior but leaves that behavior hidden in the retrieval tool. The retrieval-evidence module concentrates that behavior behind one explicit interface.
- This specification intentionally follows the completed MCP tool-turn refactor: shared tool-turn preparation remains separate from Oracle evidence ownership.
- Intended tracker triage label, if this Markdown spec is later mirrored to the project issue tracker: `ready-for-agent`.

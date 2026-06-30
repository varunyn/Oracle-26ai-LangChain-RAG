# Suggestions Observability Design

## Goal

Improve Langfuse visibility for follow-up suggestion generation without changing
suggestion behavior or adding masking. Suggestions remain independently timed
traces while being grouped with the parent chat conversation when a thread ID
is available.

## Trace model

Each `POST /api/suggestions` request creates a separate Langfuse trace named
`suggestions`. The frontend sends the current chat `thread_id`; the API uses it
as the Langfuse `session_id`. This preserves accurate suggestion latency and
failure boundaries while allowing Langfuse session replay to group suggestions
with the parent conversation.

If no thread ID is available, the trace remains standalone. A request ID is
recorded as metadata for correlation but is never used as a session ID.

## Attributes

The suggestions trace and nested observations receive:

- tags: `feature:suggestions`, `mode:suggestions`, and `model:<model-id>`
- metadata: `request_id`, `thread_id`, `mode`, `model_id`, and `outcome`
- output summary: `suggestion_count` and `outcome`

The request ID comes from the existing request context middleware. Raw prompt
content remains unchanged because this is a local non-production environment.

## Outcomes

The trace records one explicit outcome:

- `success`: one or more normalized suggestions returned
- `empty`: request was valid but no suggestions were produced
- `truncated`: the model ended because of a length limit
- `error`: generation failed

The existing API response contract remains unchanged: failures still return an
empty suggestions list.

## Implementation boundaries

- `frontend/src/hooks/useSuggestions.ts` adds the current `thread_id` to the
  suggestions request.
- `api/routes/suggestions.py` accepts the optional thread ID and passes it into
  the Langfuse trace/callback configuration.
- `src/rag_agent/utils/langfuse_tracing.py` is reused; no new tracing
  abstraction is introduced.
- Existing suggestion generation, normalization, and response schemas remain
  behaviorally unchanged.

## Verification

- Add deterministic API/unit coverage for thread ID propagation and outcome
  metadata.
- Run the focused suggestions workflow tests and Langfuse tracing unit tests.
- Trigger a real suggestion request and inspect the resulting trace with the
  Langfuse CLI, confirming session ID, tags, metadata, suggestion count, and
  outcome.


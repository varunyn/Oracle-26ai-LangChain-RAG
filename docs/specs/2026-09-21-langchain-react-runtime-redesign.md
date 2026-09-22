# LangChain React 1.1 runtime redesign specification

> **Status:** implemented — deterministic and live replay validation complete

## Problem Statement

The chat frontend uses `@langchain/react` 1.0.35, but it does not treat the library as the single owner of streamed chat state. The application combines the library's message projection with message values from graph state, reconstructs tool calls from message content when the tools projection is absent, infers a status field that the library does not expose, and intercepts thread requests to create a second transport-error channel.

Those paths were added to protect individual behaviors, but together they create competing sources of truth. A message can be deduplicated or replaced after the library has already reconciled it. A tool card can come from either the tools stream or a message-derived reconstruction. Loading and failure state can disagree across the stream, submit action, and custom fetch wrapper. Thread history is documented as server-owned, yet browser state still contains a cached history projection.

The 1.1 release adds connection lifecycle callbacks, while its resolved SDK includes fixes for root-stream failure settlement, reconnect after a clean close, rejoin behavior, persisted replay order, and later submits after a failed stream. A version-only upgrade would leave the application architecture fighting the behavior the package now provides. The upgrade should instead establish one authoritative chat runtime and remove the fallback paths it replaces.

## Solution

Upgrade the frontend to `@langchain/react` 1.1.1 and redesign the chat integration around its public projections and lifecycle.

The LangChain React stream will own messages, tool calls, graph values, loading state, stream errors, thread hydration, checkpoint metadata, and the Agent Server client. The application will add only product policy: chat mode configuration, display mapping, citation presentation, thread selection, and connection feedback.

Live and replayed messages will come from the native message projection. Live tool cards will come from the native tools projection. Because the current SDK does not seed its root tool projection when hydrating an idle thread, replayed tool cards will be hydrated at the integration boundary from the canonical `AIMessage.tool_calls` and matching `ToolMessage` records already stored in that projection. Typed graph values will carry application state such as progress, but they will not provide a second message history. The Agent Server will remain the source of truth for thread history, while browser storage will retain only the selected thread identifier.

Connection state will be explicit. The user will be able to distinguish initial thread loading, an active run, a reconnect attempt, an idle connection, and a terminal failure. The 1.1 connection callbacks will supply reconnect reason, attempt, and delay information. The UI will not infer undocumented stream fields or classify arbitrary thread HTTP responses in a custom fetch wrapper.

The redesign will not retain compatibility adapters for the old projection logic. The replay hydration boundary accepts only standard LangChain message types and stable call IDs; it does not parse message content, recognize legacy aliases, or introduce another transport or state store. If the backend does not emit the native messages or tools contract needed by the frontend, the implementation will correct that contract at its owner instead of restoring a frontend fallback.

## User Stories

1. As a chat user, I want streamed assistant text to appear once, so that I can read the response without duplicate or replaced fragments.
2. As a chat user, I want the completed answer to match the answer shown during streaming, so that completion does not rewrite visible content unexpectedly.
3. As a chat user, I want refreshed threads to replay messages in their persisted order, so that the conversation remains coherent after reload.
4. As a chat user, I want tool activity to appear while a tool runs, so that I can understand why the answer is taking time.
5. As a chat user, I want each tool call to settle as completed or failed, so that no tool card remains stuck in a running state.
6. As a chat user, I want replayed tool activity to match the original run, so that a refreshed thread tells the same story.
7. As a chat user, I want citations and source references to remain attached to the answer that produced them, so that I can verify the response after streaming or replay.
8. As a chat user, I want to see when the chat is reconnecting, so that a temporary connection loss is not mistaken for a frozen answer.
9. As a chat user, I want reconnect feedback to reflect the current attempt and wait, so that repeated retries are understandable.
10. As a chat user, I want a recovered connection to return the chat to a usable state, so that I can continue without refreshing the page.
11. As a chat user, I want a terminal connection failure to stop the loading state and show one clear recovery path, so that I am not left with a permanent spinner.
12. As a chat user, I want a later message submission to work after a failed stream has settled, so that one network failure does not poison the conversation.
13. As a chat user, I want the input and stop controls to reflect the actual run state, so that actions are enabled only when the runtime can accept them.
14. As a chat user, I want retry and regeneration to start from the correct checkpoint, so that the failed branch is replaced instead of duplicated.
15. As a chat user, I want thread switching to hydrate the selected conversation before it is shown as ready, so that messages from two threads do not flash together.
16. As a chat user, I want a newly created chat to start with an empty server-backed thread, so that prior conversation state cannot leak into it.
17. As a chat user, I want deleted chats to disappear from the sidebar and stay deleted after reload, so that browser cache cannot resurrect them.
18. As a returning chat user, I want the last selected thread to reopen, so that I can continue where I left off.
19. As a returning chat user, I want the server's thread list to determine sidebar history, so that another browser or server-side change is reflected accurately.
20. As a frontend developer, I want the graph state to have a TypeScript contract, so that invalid progress, context, reference, and message assumptions fail during the build.
21. As a frontend developer, I want one stream instance for the chat page, so that every component observes the same messages, tools, client, checkpoints, and lifecycle.
22. As a frontend developer, I want presentation mapping to consume the library's documented message and tool types, so that package upgrades do not require reverse-engineering loose record shapes.
23. As a frontend developer, I want graph values to carry only application state that is not already projected by the stream, so that two collections cannot compete to own messages.
24. As a frontend developer, I want stream failures and local operation failures to have separate, named responsibilities, so that an error is reported once by the component that can recover from it.
25. As a frontend developer, I want no custom status cast or transport-response interception, so that the integration depends only on public LangChain React contracts.
26. As a frontend developer, I want checkpoint metadata to remain the basis for retry and fork actions, so that branch operations follow persisted Agent Server state.
27. As a frontend developer, I want old reconciliation and fallback tests removed, so that the test suite protects the intended runtime instead of preserving retired behavior.
28. As an operator, I want reconnect and terminal failure behavior to be reproducible in a browser test, so that deployment networking changes cannot silently freeze chat.
29. As an operator, I want live verification against the Agent Server before release, so that deterministic mocks do not hide a protocol mismatch.
30. As a maintainer, I want the chat streaming documentation to name one owner for each piece of state, so that future changes are added at the right boundary.

## Implementation Decisions

- Upgrade `@langchain/react` from 1.0.35 to 1.1.1 and accept the compatible `@langchain/langgraph-sdk` version resolved by that release. React 19 and the installed LangChain Core line already satisfy the published peer ranges.
- Define one TypeScript graph-state contract that mirrors the public chat graph state used by the browser. It will cover messages and application values such as progress, context, and references without exposing backend-only working state.
- Instantiate the LangChain React stream once at the page provider boundary. Use the package's stream provider and context access for child components. Keep product configuration and connection presentation in a small application-owned layer rather than wrapping or duplicating stream state.
- Treat the native messages projection as the only frontend message timeline for live output and replay. Remove reconciliation against messages read from graph values, optimistic echo heuristics, generated-ID replacement, and live-versus-finalized selection logic.
- Treat the native tools projection as the only source for live tool cards. Map the documented assembled tool-call fields directly to the product view model. For idle thread replay, seed the root presentation projection from typed `AIMessage.tool_calls` and matching `ToolMessage.tool_call_id` records because the SDK currently performs that checkpoint seeding only for scoped projections. Live lifecycle state replaces the replay seed by call ID. Remove message-content reconstruction and loose legacy field aliases.
- Use graph values only for application state that the stream does not already project, including progress data required by the current progress UI.
- Preserve message metadata lookup for checkpoint-aware retry, regeneration, and fork behavior.
- Replace inferred status with an explicit product lifecycle. The lifecycle states are hydrating, idle, running, reconnecting, and failed. Thread loading, run loading, stream error, and the 1.1 connection callbacks supply the transitions.
- Use the 1.1 connection callbacks for initial connection and reconnect feedback. Reconnect state records the reason, attempt number, and delay supplied by the package. A successful connection clears reconnect state.
- Use the root stream error as the authoritative stream failure. Submit actions may retain short-lived validation or action errors when the failure occurs before a stream begins. Thread delete, cancel, and history refresh errors remain local to those operations.
- Remove the custom fetch interception used to infer transport errors from thread endpoints. Authentication or request customization may use the supported stream configuration, but it must not create a parallel lifecycle model.
- Keep the Agent Server as the source of truth for thread history. Browser storage retains only the active thread identifier. The sidebar is replaced from server search results and never restored from a cached browser history.
- Preserve the existing user-facing contracts for chat modes, citations, references, feedback, copy actions, stop, retry, regeneration, and thread deletion.
- Correct the backend or stream-mode contract if native messages, tools, metadata, or replay data are missing. Keep the SDK root-hydration adapter narrow, typed, and deletable once upstream seeds the root tool projection; do not add another fallback or compatibility path.
- Delete retired adapters, status normalization, fallback branches, and their tests in the same delivery. No legacy path remains after the migration.
- Update the chat streaming and memory documentation when implementation details change. Record the completed redesign in the changelog.

## Testing Decisions

- The main acceptance seam is the existing browser chat flow speaking the Agent Server protocol. A deterministic protocol fixture will drive thread hydration, streamed messages, native tool events, stream completion, replay, reconnect, terminal failure, later submit, retry, thread switching, and deletion through the same UI a user operates.
- Browser assertions will cover visible and semantic behavior. They will not assert internal hook calls, reducer steps, private package fields, or the number of React renders.
- The connection suite will simulate a dropped root stream, a clean server close, a successful reconnect, an exhausted reconnect path, and a new submit after settlement. It will prove that loading ends, status changes are visible, and the chat remains usable after recovery.
- The message suite will prove that token streaming, final completion, and persisted replay produce one ordered conversation without duplicate user or assistant messages.
- The tool suite will use native tools-channel events for live calls and standard persisted AI/tool messages for idle replay. It will prove running, completed, failed, and replayed cards without historical tools-channel events or extraction from message content.
- The thread suite will prove that local storage contains only the selected thread identifier, server search replaces sidebar history, hydration prevents cross-thread flashes, and deletion survives reload.
- Focused unit tests will remain for pure conversion boundaries such as message presentation, references, and direct assembled-tool-call mapping. They will use documented LangChain types and will not preserve fallback input shapes.
- Existing frontend build, lint, unit, dead-code, and deterministic end-to-end checks remain required. Dead-code analysis must prove that retired reconciliation helpers and compatibility branches are gone.
- A live Agent Server and frontend path is required before calling the implementation complete. It will exercise direct chat, a tool-producing mode, persisted replay, retry, citations, connection recovery where the environment permits it, and a later submit. If infrastructure prevents a live check, the implementation report must name the unproven behavior instead of treating deterministic tests as equivalent evidence.
- The highest test seam is expected to carry most of the behavior. New lower-level seams should be added only when they make a pure mapping failure easier to diagnose.

## Out of Scope

- Subagent rendering and subagent-specific event timelines.
- Human-in-the-loop interrupt approval interfaces.
- Queued message submission while another run is active.
- Streamed media, multimodal content, or headless tool rendering.
- Custom stream channels beyond the current messages, tools, and application progress needs.
- A broader redesign of message visuals, tool-card visuals, citations, or sidebar styling.
- Backend graph behavior changes unrelated to emitting the documented native frontend contract.
- Compatibility with the retired projection, status, content-derived tool reconstruction, transport-error, or browser-history cache paths.

## Further Notes

- The redesign follows the project's existing rule that the Agent Server owns thread state and the browser stores only a reopen pointer. Where the implementation differs from that documentation, the implementation changes.
- The 1.1 release makes connection lifecycle observable through `onConnected` and the expanded `onReconnect` payload. The resolved SDK also fixes several failure and rejoin cases that the current custom state tries to approximate.
- The useful package features for this delivery are typed stream state, native messages and tool-call projections, thread hydration and loading state, root stream errors, message checkpoint metadata, and connection lifecycle callbacks.
- Live browser verification selected an existing Agent Server thread with three persisted tool calls and confirmed that all three completed cards and their outputs replayed after the Docker frontend rebuild.
- Submission queues, interrupt hooks, custom channel hooks, subagent projections, and streamed media are useful library capabilities, but this application does not need them to complete the runtime redesign.
- Primary references are the official [`@langchain/react` changelog](https://github.com/langchain-ai/langgraphjs/blob/main/libs/sdk-react/CHANGELOG.md), the official [frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview), and the official [join and rejoin guide](https://docs.langchain.com/oss/javascript/langchain/frontend/join-rejoin).

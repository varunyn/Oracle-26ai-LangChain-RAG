# Frontend AGENTS.md

This guide applies only to `frontend/`. The app is a Next.js 16 / React 19 / TypeScript / Tailwind CSS v4 client.

## Next.js: ALWAYS read docs before coding

# Frontend — AGENTS.md (scoped to frontend/)

Before any Next.js work, find and read the relevant doc in `node_modules/next/dist/docs/`. Your training data is outdated — the docs are the source of truth.

## LangChain Frontend Ownership

Before changing chat behavior, verify the current official LangChain frontend docs for the relevant pattern.

- `@langchain/react` is the frontend chat runtime. It owns `useStream`, selector hooks such as `useChannel`, thread/run lifecycle, stream transport, `stream.messages`, `stream.values`, and `stream.toolCalls`.
- `@langchain/core` is for message classes/types and test fixtures only. Do not build a second frontend chat runtime or state model around it.
- AI Elements is the presentation layer. `Conversation`, `Message`, `Tool`, suggestions, and related components should receive already-derived state and render it; they should not own chat transport or event parsing.
- Prefer native `@langchain/react` surfaces before custom glue:
  - messages: `stream.messages`
  - tool calls: `stream.toolCalls`
  - finalized state: `stream.values`
  - custom backend channels: `useChannel(...)`
- If the backend cannot emit native tool-call events and must expose a custom channel such as `custom:mcp_tool_activity`, keep exactly one adapter close to the stream provider and feed its output into presentational components. Do not spread custom event parsing across provider, controller, and UI layers.

## Mode/Channel Contract

- `direct`: native message stream only; no MCP activity channel
- `rag`: native message/state stream plus citations; retrieval is not MCP activity
- `mcp`: MCP activity uses `custom:mcp_tool_activity` when an MCP tool actually runs
- `mixed`: retrieval/citations use the normal state/message contract; MCP activity uses `custom:mcp_tool_activity` only when an MCP tool actually runs

- `stream.messages`: live model-message streams and message content deltas
- `stream.values`: graph-state snapshots during a run
- hydrated LangGraph thread state: authoritative replay/final message state
- `stream.toolCalls`: native LangGraph/LangChain tool execution lifecycle only
- `custom:mcp_tool_activity`: MCP-specific lifecycle progress, consumed once by the stream provider

Do not infer server identity by splitting a tool name. The backend must send the configured server key when it can resolve one.

- Chat streams come directly from the LangGraph Agent Server at `NEXT_PUBLIC_LANGGRAPH_API_BASE` using assistant id `chat_agent`; product/config/document calls use `NEXT_PUBLIC_API_BASE`.
- Preserve optimistic message ids, server thread ids, structured content blocks, replay deduplication, citations, and visible tool progress when changing chat state or rendering.
- MCP server configuration and observability links are edited through `src/app/settings/page.tsx` and the FastAPI config APIs.

COMMANDS (pnpm required)

- Install: pnpm install
- Dev: PORT=4000 pnpm dev (local dev on http://localhost:4000)
- Build: pnpm build (type/lint checks gate the build)
- Lint: pnpm lint
- Unit: pnpm test
- E2E: pnpm test:e2e

ANTI-PATTERNS (FORBIDDEN)

- Using npm/yarn; this repo mandates pnpm for Next.js
- Importing CSS/tailwind outside top-level layout files
- Hardcoding backend URLs; use env configuration
- Changing Agent Server stream/event expectations in UI without updating graph/workflow tests
- Adding a second frontend chat transport or proxy when the configured Agent Server transport already covers the path
- Building a second message/tool state store beside `@langchain/react` without proving a backend contract gap first
- Treating AI Elements components as state managers instead of renderers
- Mixing native `stream.toolCalls` and scattered custom tool-event parsing in multiple layers
- Adding fallback or legacy chat code paths when one authoritative runtime path can be made correct
- Do not create a separate channel contract per mode
- Do not route RAG retrieval through custom tool-activity channels
- Do not convert MCP activity into fake native `stream.toolCalls`
- Do not treat `stream.messages` or an intermediate `stream.values` snapshot as replay authority

NOTES

- See root AGENTS.md for repo-wide rules and command matrix
- See api/AGENTS.md for backend streaming protocol invariants
- See `src/AGENTS.md` for graph/runtime ownership and LangChain documentation requirements

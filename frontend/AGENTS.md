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
  - custom backend channels: `useChannel(...)` only for non-tool, non-message, non-state data that cannot be represented by native projections

ANTI-PATTERNS (FORBIDDEN)

- Hardcoding backend URLs; use env configuration
- Adding fallback or legacy chat code paths when one authoritative runtime path can be made correct

  NOTES

- See root AGENTS.md for repo-wide rules and command matrix
- See `src/AGENTS.md` for graph/runtime ownership and LangChain documentation requirements

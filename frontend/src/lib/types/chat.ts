/**
 * Shared chat-related types
 * Extracted from page.tsx for reuse across components and utilities
 */

/** Segment of message content: either plain text or a citation marker index */
export type ContentSegment =
  | { type: "text"; content: string }
  | { type: "citation"; index: number };

/** References attached to an assistant message (citations, reranker docs, MCP, errors) */
export type MessageReferences = {
  standalone_question?: string;
  citations: { source: string; page: string }[];
  reranker_docs: {
    page_content: string;
    metadata: Record<string, unknown>;
  }[];
  mcp_used?: boolean;
  mcp_tools_used?: string[];
  error?: string;
};

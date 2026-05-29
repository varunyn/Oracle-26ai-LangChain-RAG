/**
 * Shared chat-related types
 * Extracted from page.tsx for reuse across components and utilities
 */

/** Segment of message content: either plain text or a citation marker index */
export type ContentSegment =
  | { type: "text"; content: string }
  | { type: "citation"; index: number };

/** One MCP tool execution (args + tool result text), in conversation order */
export type McpToolInvocation = {
  tool_name: string;
  args?: unknown;
  result?: string | null;
  error?: string | null;
};

/** Streaming MCP tool lifecycle event emitted while a run is in progress */
export type McpProgressEvent = {
  phase: "start" | "end" | "error";
  tool_name: string;
  tool_run_id?: string;
  args?: unknown;
  result?: string | null;
  error?: string | null;
};

/** Runtime context window usage for the current answer */
export type ContextUsage = {
  tokens: number;
  max: number;
  percent: number;
  model_id?: string;
};

/** References attached to an assistant message (citations, reranker docs, MCP, errors) */
export type MessageReferences = {
  trace_id?: string;
  standalone_question?: string;
  citations: { source: string; page: string }[];
  reranker_docs: {
    page_content: string;
    metadata: Record<string, unknown>;
  }[];
  mcp_used?: boolean;
  mcp_tools_used?: string[];
  context_usage?: ContextUsage;
  /** Populated when the runtime returns per-call args/results from the agent */
  mcp_tool_invocations?: McpToolInvocation[];
  /** Streaming MCP tool progress while the run is still in-flight */
  mcp_progress_events?: McpProgressEvent[];
  error?: string;
};

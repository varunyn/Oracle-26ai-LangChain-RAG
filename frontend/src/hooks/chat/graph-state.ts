import type { BaseMessage } from "@langchain/core/messages";

/** Public graph state emitted to the browser by the chat agent. */
export interface PublicChatGraphState extends Record<string, unknown> {
  context?: {
    collection_name?: string;
    mode?: "direct" | "rag" | "mcp" | "mixed";
  };
  messages?: BaseMessage[];
  progress?: string;
  references?: Record<string, unknown>;
}

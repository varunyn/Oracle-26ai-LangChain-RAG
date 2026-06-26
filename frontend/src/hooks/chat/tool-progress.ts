import type { AssembledToolCall } from "@langchain/react";
import type {
  McpProgressEvent,
} from "@/lib/types/chat";
import type { MessageLike, ReferencePayload } from "@/hooks/chat/controller-types";
import { referencePayloadFromMessage } from "@/hooks/chat/references";

export type SdkToolProgress = {
  toolCallId?: string;
  name?: string;
  state?: string;
  input?: unknown;
  data?: unknown;
  result?: unknown;
  error?: unknown;
};

export function stringifyToolPayload(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function toolCallsToMcpEvents(toolCalls: AssembledToolCall[]): McpProgressEvent[] {
  return toolCalls
    .map((tool): McpProgressEvent | null => {
      const toolName = typeof tool.name === "string" ? tool.name.trim() : "";
      if (!toolName) return null;
      const toolRunId =
        typeof tool.callId === "string" && tool.callId.trim() ? tool.callId : undefined;
      const base = {
        tool_name: toolName,
        tool_run_id: toolRunId,
        args: tool.input ?? tool.args,
      };
      if (tool.status === "finished") {
        return { ...base, phase: "end", result: stringifyToolPayload(tool.output) };
      }
      if (tool.status === "error") {
        return { ...base, phase: "error", error: stringifyToolPayload(tool.error) };
      }
      return { ...base, phase: "start", result: null };
    })
    .filter((event): event is McpProgressEvent => event != null);
}

export function withLiveToolProgress(
  messages: MessageLike[],
  progressEvents: McpProgressEvent[],
): MessageLike[] {
  if (progressEvents.length === 0) return messages;
  const toolsUsed = [...new Set(progressEvents.map((event) => event.tool_name))];
  const assistantIndex = [...messages]
    .map((message, index) => ({ message, index }))
    .reverse()
    .find(({ message }) => message.role === "assistant")?.index;
  const progressReferences = {
    citations: [],
    reranker_docs: [],
    mcp_used: true,
    mcp_tools_used: toolsUsed,
    mcp_progress_events: progressEvents,
  } satisfies ReferencePayload;

  if (assistantIndex == null) {
    const latestEvent = progressEvents.at(-1);
    return [
      ...messages,
      {
        id: `live-tool-progress-${latestEvent?.tool_run_id ?? latestEvent?.tool_name ?? "tool"}`,
        role: "assistant",
        content: "",
        references: progressReferences,
      },
    ];
  }

  const target = messages[assistantIndex];
  const currentReferences = referencePayloadFromMessage(target);
  const mergedReferences: ReferencePayload = {
    trace_id: currentReferences?.trace_id,
    standalone_question: currentReferences?.standalone_question,
    citations: currentReferences?.citations ?? [],
    reranker_docs: currentReferences?.reranker_docs ?? [],
    context_usage: currentReferences?.context_usage,
    mcp_used: currentReferences?.mcp_used ?? true,
    mcp_tools_used:
      currentReferences?.mcp_tools_used && currentReferences.mcp_tools_used.length > 0
        ? currentReferences.mcp_tools_used
        : toolsUsed,
    mcp_tool_invocations: currentReferences?.mcp_tool_invocations,
    mcp_progress_events: progressEvents,
    error: currentReferences?.error,
  };
  const next = [...messages];
  next[assistantIndex] = {
    ...target,
    references: mergedReferences,
  };
  return next;
}

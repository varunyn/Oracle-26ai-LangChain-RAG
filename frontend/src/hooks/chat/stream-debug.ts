import type { MessageLike, ReferencePayload } from "@/hooks/chat/controller-types";
import {
  type BaseMessageWithKwargs,
  readText,
  toReferences,
  toRole,
} from "@/hooks/chat/references";
import type { SdkToolProgress } from "@/hooks/chat/tool-progress";
import { getMessageContent } from "@/lib/chat/messages";

export type ChatStreamDebugEvent =
  | "submit"
  | "stop"
  | "error"
  | "status"
  | "stream.messages"
  | "stream.toolProgress"
  | "visible.messages";

const CHAT_STREAM_DEBUG_FLAG = "rag_agent_debug_stream";
const CHAT_STREAM_DEBUG_BREAK_FLAG = "rag_agent_debug_stream_break";

function getDebugParam(name: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return new URLSearchParams(window.location.search).get(name);
  } catch {
    return null;
  }
}

function getDebugStorageValue(name: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(name);
  } catch {
    return null;
  }
}

function isTruthyDebugValue(value: string | null | undefined): boolean {
  if (!value) return false;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

export function isChatStreamDebugEnabled(): boolean {
  return (
    process.env.NEXT_PUBLIC_CHAT_STREAM_DEBUG === "true" ||
    isTruthyDebugValue(getDebugParam("debugStream")) ||
    isTruthyDebugValue(getDebugStorageValue(CHAT_STREAM_DEBUG_FLAG))
  );
}

function shouldBreakForChatStreamEvent(event: ChatStreamDebugEvent): boolean {
  const configured = getDebugParam("debugStreamBreak") ?? getDebugStorageValue(CHAT_STREAM_DEBUG_BREAK_FLAG);
  if (!configured) return false;
  const normalized = configured.trim().toLowerCase();
  if (["1", "true", "yes", "on", "*", "all"].includes(normalized)) {
    return true;
  }
  return normalized
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .includes(event.toLowerCase());
}

function summarizeReferencePayload(refs: ReferencePayload | null): Record<string, unknown> | null {
  if (!refs) return null;
  return {
    trace_id: refs.trace_id,
    mcp_used: refs.mcp_used,
    mcp_tools_used: refs.mcp_tools_used,
    mcp_tool_invocations: refs.mcp_tool_invocations?.map((tool) => tool.tool_name),
    mcp_progress_events: refs.mcp_progress_events?.map((event) => ({
      phase: event.phase,
      tool_name: event.tool_name,
      tool_run_id: event.tool_run_id,
    })),
    citations: refs.citations?.length ?? 0,
    reranker_docs: refs.reranker_docs?.length ?? 0,
    error: refs.error,
  };
}

export function summarizeBaseMessage(
  message: BaseMessageWithKwargs,
  index: number,
): Record<string, unknown> {
  return {
    index,
    id: typeof message.id === "string" ? message.id : undefined,
    role: toRole(message),
    text: readText(message.content).slice(0, 240),
    refs: summarizeReferencePayload(toReferences(message)),
  };
}

export function summarizeVisibleMessage(message: MessageLike, index: number): Record<string, unknown> {
  return {
    index,
    id: message.id,
    role: message.role,
    text: getMessageContent(message).slice(0, 240),
    refs: summarizeReferencePayload(message.references ?? null),
  };
}

export function summarizeToolProgress(toolProgress: SdkToolProgress[]): Record<string, unknown>[] {
  return toolProgress.map((tool, index) => ({
    index,
    toolCallId: tool.toolCallId,
    name: tool.name,
    state: tool.state,
    input: tool.input,
    data: tool.data,
    result: tool.result,
    error: tool.error,
  }));
}

export function debugChatStream(
  event: ChatStreamDebugEvent,
  payload: Record<string, unknown>,
): void {
  if (!isChatStreamDebugEnabled()) return;
  const time = new Date().toISOString();
  console.groupCollapsed(`[chat-stream] ${event} ${time}`);
  console.log(payload);
  console.groupEnd();
  if (shouldBreakForChatStreamEvent(event)) {
    debugger;
  }
}

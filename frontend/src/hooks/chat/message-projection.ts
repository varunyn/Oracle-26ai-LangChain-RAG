import type { ChatStatus, MessageLike } from "@/hooks/chat/controller-types";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";
import {
  type BaseMessageWithKwargs,
  toReferences,
  toRole,
} from "@/hooks/chat/references";
import { getMessageContent } from "@/lib/chat/messages";

interface ToolCallLike {
  args?: unknown;
  id?: unknown;
  name?: unknown;
}

interface ToolCallContainer {
  tool_calls?: ToolCallLike[];
  toolCalls?: ToolCallLike[];
}

function isToolMessage(message: BaseMessageWithKwargs): boolean {
  const serialized = message as BaseMessageWithKwargs & {
    role?: unknown;
    type?: unknown;
  };
  const role =
    typeof serialized.role === "string" ? serialized.role.toLowerCase() : "";
  const type =
    typeof serialized.type === "string" ? serialized.type.toLowerCase() : "";
  return role === "tool" || type === "tool";
}

function toToolCallIds(message: BaseMessageWithKwargs): string[] | undefined {
  const serialized = message as BaseMessageWithKwargs & ToolCallContainer;
  const toolCalls = serialized.tool_calls ?? serialized.toolCalls;
  const ids = toolCalls
    ?.map((toolCall) => toolCall.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  return ids && ids.length > 0 ? ids : undefined;
}

function replaceReplayedMessagesById(messages: MessageLike[]): MessageLike[] {
  const seenIds = new Map<string, number>();
  const normalized: MessageLike[] = [];

  for (const message of messages) {
    const id = typeof message.id === "string" ? message.id.trim() : "";
    if (id) {
      const existingIndex = seenIds.get(id);
      if (existingIndex != null) {
        normalized[existingIndex] = message;
        continue;
      }
      seenIds.set(id, normalized.length);
    }
    normalized.push(message);
  }

  return normalized;
}

export function normalizeStatus(
  rawStatus: unknown,
  isLoading: boolean,
  hasError: boolean
): ChatStatus {
  if (hasError) {
    return "error";
  }
  if (
    rawStatus === "submitted" ||
    rawStatus === "streaming" ||
    rawStatus === "ready" ||
    rawStatus === "error"
  ) {
    return rawStatus;
  }
  return isLoading ? "streaming" : "ready";
}

export function getLastUserMessageText(messages: MessageLike[]): string {
  const lastUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  if (lastUserMessage == null) {
    return "";
  }
  return getMessageContent(lastUserMessage).trim();
}

export function projectStreamMessages(args: {
  streamMessages: BaseMessageWithKwargs[] | undefined;
}): MessageLike[] {
  const { streamMessages } = args;
  const mapped = (streamMessages ?? [])
    .filter((message) => !isToolMessage(message))
    .map((message, index) => {
      const toolCallIds = toToolCallIds(message);
      const content = getMessageContent(message);
      const displayContent =
        toolCallIds && content.trim() === "." ? "" : content;
      return {
        id: typeof message.id === "string" ? message.id : `message-${index}`,
        role: toRole(message),
        content: displayContent,
        ...(toolCallIds ? { toolCallIds } : {}),
        references: toReferences(message),
      };
    });
  const normalized = replaceReplayedMessagesById(mapped);
  debugChatStage("projectStreamMessages", {
    rawCount: streamMessages?.length ?? 0,
    mapped: summarizeMessages(mapped),
    normalized: summarizeMessages(normalized),
    projected: summarizeMessages(normalized),
  });
  return normalized;
}

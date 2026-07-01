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

function isEphemeralStreamMessageId(id: string): boolean {
  return id.startsWith("lc_run--");
}

function removeDuplicateAssistantProjections(
  messages: MessageLike[]
): MessageLike[] {
  const result: MessageLike[] = [];

  for (const message of messages) {
    if (message.role !== "assistant") {
      result.push(message);
      continue;
    }

    const content = message.content?.trim() ?? "";
    const messageId = typeof message.id === "string" ? message.id.trim() : "";
    const duplicateIndex = result.findIndex(
      (existing) =>
        existing.role === "assistant" &&
        (existing.content?.trim() ?? "") === content &&
        typeof existing.id === "string" &&
        (isEphemeralStreamMessageId(existing.id) ||
          isEphemeralStreamMessageId(messageId))
    );

    if (duplicateIndex === -1) {
      result.push(message);
      continue;
    }

    const existing = result[duplicateIndex];
    if (
      typeof existing.id === "string" &&
      isEphemeralStreamMessageId(existing.id) &&
      !isEphemeralStreamMessageId(messageId)
    ) {
      result[duplicateIndex] = message;
    }
  }

  return result;
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
  const normalized = removeDuplicateAssistantProjections(
    replaceReplayedMessagesById(mapped)
  );
  debugChatStage("projectStreamMessages", {
    rawCount: streamMessages?.length ?? 0,
    mapped: summarizeMessages(mapped),
    normalized: summarizeMessages(normalized),
    projected: summarizeMessages(normalized),
  });
  return normalized;
}

function preferContent(
  primary: string | undefined,
  secondary: string | undefined
): string | undefined {
  const primaryText = primary?.trim() ?? "";
  const secondaryText = secondary?.trim() ?? "";
  if (!primaryText) {
    return secondary;
  }
  if (!secondaryText) {
    return primary;
  }
  return secondaryText.length >= primaryText.length ? secondary : primary;
}

function preferToolCallIds(
  primary: string[] | undefined,
  secondary: string[] | undefined
): string[] | undefined {
  if (!primary?.length) {
    return secondary;
  }
  if (!secondary?.length) {
    return primary;
  }
  return secondary.length >= primary.length ? secondary : primary;
}

export function mergeProjectedMessages(
  primary: MessageLike[],
  secondary: MessageLike[]
): MessageLike[] {
  if (primary.length === 0) {
    return secondary;
  }
  if (secondary.length === 0) {
    return primary;
  }

  const merged = [...primary];
  const indexById = new Map<string, number>();
  for (const [index, message] of merged.entries()) {
    const id = typeof message.id === "string" ? message.id.trim() : "";
    if (id) {
      indexById.set(id, index);
    }
  }

  for (const message of secondary) {
    const id = typeof message.id === "string" ? message.id.trim() : "";
    if (!id) {
      merged.push(message);
      continue;
    }
    const existingIndex = indexById.get(id);
    if (existingIndex == null) {
      indexById.set(id, merged.length);
      merged.push(message);
      continue;
    }
    const existing = merged[existingIndex];
    merged[existingIndex] = {
      ...existing,
      ...message,
      role: existing.role ?? message.role,
      content: preferContent(existing.content, message.content),
      toolCallIds: preferToolCallIds(existing.toolCallIds, message.toolCallIds),
      references: existing.references ?? message.references,
    };
  }
  return merged;
}

export function selectMessagesForStatus(
  liveMessages: MessageLike[],
  finalizedMessages: MessageLike[] | undefined,
  status: ChatStatus
): MessageLike[] {
  if (status === "submitted" || status === "streaming") {
    return liveMessages;
  }
  if (status === "ready" && finalizedMessages !== undefined) {
    return finalizedMessages;
  }
  return mergeProjectedMessages(liveMessages, finalizedMessages ?? []);
}

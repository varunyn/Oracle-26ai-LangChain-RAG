import { getMessageContent } from "@/lib/chat/messages";
import type {
  ChatStatus,
  MessageLike,
} from "@/hooks/chat/controller-types";
import {
  type BaseMessageWithKwargs,
  toReferences,
  toRole,
} from "@/hooks/chat/references";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";

type ToolCallLike = {
  id?: unknown;
};

function preferContent(primary: string | undefined, secondary: string | undefined): string | undefined {
  const primaryText = primary?.trim() ?? "";
  const secondaryText = secondary?.trim() ?? "";
  if (!primaryText) return secondary;
  if (!secondaryText) return primary;
  return secondaryText.length >= primaryText.length ? secondary : primary;
}

function preferToolCallIds(
  primary: string[] | undefined,
  secondary: string[] | undefined,
): string[] | undefined {
  if (!primary?.length) return secondary;
  if (!secondary?.length) return primary;
  return secondary.length >= primary.length ? secondary : primary;
}

function toToolCallIds(message: BaseMessageWithKwargs): string[] | undefined {
  const toolCalls = (message as BaseMessageWithKwargs & { tool_calls?: ToolCallLike[] }).tool_calls;
  const ids = toolCalls
    ?.map((toolCall) => toolCall.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  return ids && ids.length > 0 ? ids : undefined;
}

function dedupeProjectedMessages(messages: MessageLike[]): MessageLike[] {
  const seenIds = new Map<string, number>();
  const deduped: MessageLike[] = [];

  for (const message of messages) {
    const id = typeof message.id === "string" ? message.id.trim() : "";
    if (id) {
      const existingIndex = seenIds.get(id);
      if (existingIndex != null) {
        deduped[existingIndex] = message;
        continue;
      }
      seenIds.set(id, deduped.length);
    }
    deduped.push(message);
  }

  return deduped;
}

export function normalizeStatus(
  rawStatus: unknown,
  isLoading: boolean,
  hasError: boolean,
): ChatStatus {
  if (hasError) return "error";
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
  const lastUserMessage = [...messages].reverse().find((message) => message.role === "user");
  if (lastUserMessage == null) return "";
  return getMessageContent(lastUserMessage).trim();
}

export function projectStreamMessages(args: {
  streamMessages: BaseMessageWithKwargs[] | undefined;
}): MessageLike[] {
  const { streamMessages } = args;
  const mapped = (streamMessages ?? []).map((message, index) => {
    const toolCallIds = toToolCallIds(message);
    return {
      id: typeof message.id === "string" ? message.id : `message-${index}`,
      role: toRole(message),
      content: getMessageContent(message),
      ...(toolCallIds ? { toolCallIds } : {}),
      references: toReferences(message),
    };
  });
  const deduped = dedupeProjectedMessages(mapped);
  debugChatStage("projectStreamMessages", {
    rawCount: streamMessages?.length ?? 0,
    mapped: summarizeMessages(mapped),
    deduped: summarizeMessages(deduped),
    projected: summarizeMessages(deduped),
  });
  return deduped;
}

export function mergeProjectedMessages(
  primary: MessageLike[],
  secondary: MessageLike[],
): MessageLike[] {
  if (primary.length === 0) return secondary;
  if (secondary.length === 0) return primary;

  const merged: MessageLike[] = [...primary];
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

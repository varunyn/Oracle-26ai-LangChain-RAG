import { getMessageContent } from "@/lib/chat/messages";
import type { McpProgressEvent } from "@/lib/types/chat";
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
import { withLiveToolProgress } from "@/hooks/chat/tool-progress";

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
  liveToolProgressEvents: McpProgressEvent[];
}): MessageLike[] {
  const { streamMessages, liveToolProgressEvents } = args;
  const mapped = (streamMessages ?? []).map((message, index) => ({
    id: typeof message.id === "string" ? message.id : `message-${index}`,
    role: toRole(message),
    content: getMessageContent(message),
    references: toReferences(message),
  }));
  const deduped = dedupeProjectedMessages(mapped);
  const projected = withLiveToolProgress(deduped, liveToolProgressEvents);
  debugChatStage("projectStreamMessages", {
    rawCount: streamMessages?.length ?? 0,
    mapped: summarizeMessages(mapped),
    deduped: summarizeMessages(deduped),
    projected: summarizeMessages(projected),
    liveToolProgressCount: liveToolProgressEvents.length,
  });
  return projected;
}

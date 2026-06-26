import { getMessageContent } from "@/lib/chat/messages";
import type { McpProgressEvent } from "@/lib/types/chat";
import type {
  ChatStatus,
  MessageLike,
} from "@/hooks/chat/controller-types";
import {
  type BaseMessageWithKwargs,
  readText,
  toReferences,
  toRole,
} from "@/hooks/chat/references";
import { withLiveToolProgress } from "@/hooks/chat/tool-progress";

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
    content: readText(message.content),
    references: toReferences(message),
  }));

  return withLiveToolProgress(mapped, liveToolProgressEvents);
}

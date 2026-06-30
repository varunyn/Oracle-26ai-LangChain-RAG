import type { FlowMode } from "@/hooks/useChatBodyParams";
import type { ContextUsage, MessageReferences } from "@/lib/types/chat";

export type ToastApi = {
  error: (description: string, title?: string) => void;
  success: (description: string, title?: string) => void;
};

export type ReferencePayload = MessageReferences;

export type MessageLike = {
  id?: string;
  role?: string;
  content?: string;
  toolCallIds?: string[];
  references?: ReferencePayload | null;
};

export type ChatStatus = "submitted" | "streaming" | "ready" | "error";

export type SendOverrides = {
  mode?: FlowMode;
};

export type ClearSessionChat = (helpers: {
  threadId?: string | null;
  setMessages?: (
    value: MessageLike[] | ((prev: MessageLike[]) => MessageLike[])
  ) => void;
  setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
  setContextUsage: (
    value:
      | ContextUsage
      | null
      | ((prev: ContextUsage | null) => ContextUsage | null)
  ) => void;
}) => void;

export type RemoveThreadHistoryEntry = (threadId: string) => void;

export type UseChatControllerArgs = {
  selectedModel: string;
  threadId: string | null;
  sessionId: string;
  collectionName: string;
  enableReranker: boolean;
  enableTracing: boolean;
  flowMode: FlowMode;
  toast: ToastApi;
  clearSessionChat: ClearSessionChat;
  removeThreadHistoryEntry: RemoveThreadHistoryEntry;
};

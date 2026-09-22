import type { FlowMode } from "@/hooks/useChatBodyParams";
import type { ThreadHistoryClient } from "@/hooks/useChatSession";
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

export type ChatStatus =
  | "hydrating"
  | "idle"
  | "running"
  | "reconnecting"
  | "failed";

export type SendOverrides = {
  forkFromCheckpointId?: string;
  mode?: FlowMode;
};

export type ClearSessionChat = (helpers: {
  threadId?: string | null;
  setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
  setContextUsage: (
    value:
      | ContextUsage
      | null
      | ((prev: ContextUsage | null) => ContextUsage | null)
  ) => void;
}) => void;

export type RefreshThreadHistory = (
  client: ThreadHistoryClient
) => Promise<void>;

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
  refreshThreadHistory: RefreshThreadHistory;
};

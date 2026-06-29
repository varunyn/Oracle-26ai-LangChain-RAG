type MessageSummary = {
  index: number;
  id: string;
  role: string;
  preview: string;
};

type MessageLike = {
  id?: string;
  role?: string;
  content?: unknown;
};

type MessageCollectionSummary = {
  total: number;
  duplicateIds: string[];
  messages: MessageSummary[];
};

const PREVIEW_LIMIT = 80;

function normalizePreview(content: unknown): string {
  if (typeof content !== "string") return "";
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= PREVIEW_LIMIT) return normalized;
  return `${normalized.slice(0, PREVIEW_LIMIT - 3)}...`;
}

export function summarizeMessages(messages: MessageLike[]): MessageCollectionSummary {
  const idCounts = new Map<string, number>();
  const summaryMessages = messages.map((message, index) => {
    const id = typeof message.id === "string" ? message.id : `message-${index}`;
    idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
    return {
      index,
      id,
      role: typeof message.role === "string" ? message.role : "unknown",
      preview: normalizePreview(message.content),
    };
  });

  return {
    total: messages.length,
    duplicateIds: [...idCounts.entries()]
      .filter(([, count]) => count > 1)
      .map(([id]) => id),
    messages: summaryMessages,
  };
}

export function debugChatStage(stage: string, details: Record<string, unknown>): void {
  if (process.env.NODE_ENV === "production") return;
  console.debug(`[chat-debug] ${stage}`, details);
}

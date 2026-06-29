import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { THREAD_ID_STORAGE_KEY } from "@/constants/chat";
import { debugChatStage } from "@/hooks/chat/debug";

const MAX_STORED_THREADS = 30;
const CHAT_SESSION_EVENT = "rag-agent-chat-session-change";

export type ChatThreadSummary = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
};

export type SessionState = {
  threadId: string | null;
  threadHistory: ChatThreadSummary[];
  hydrated: boolean;
};

type ThreadSearchResult = {
  thread_id: string;
  created_at: string;
  updated_at: string;
  values?: {
    messages?: Array<Record<string, unknown>>;
  };
};

type ThreadHistoryClient = {
  threads: {
    search: (query: {
      limit: number;
      sortBy: "updated_at";
      sortOrder: "desc";
      select: ["thread_id", "created_at", "updated_at", "values"];
    }) => Promise<ThreadSearchResult[]>;
  };
};

const EMPTY_SESSION_STATE: SessionState = {
  threadId: null,
  threadHistory: [],
  hydrated: false,
};
let cachedSessionSnapshot: SessionState | null = null;

/** Session ID: new per tab load/refresh (not persisted). Used for Langfuse session grouping. */
function generateSessionId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}

function defaultTitle(threadId: string): string {
  return `Chat ${threadId.slice(-6)}`;
}

function readStoredThreadId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const storedThreadId = window.localStorage.getItem(THREAD_ID_STORAGE_KEY);
    return storedThreadId?.trim() || null;
  } catch {
    return null;
  }
}

function parseTimestamp(value: string, defaultValue: number): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

function deriveThreadTitle(thread: ThreadSearchResult): string {
  const messages = Array.isArray(thread.values?.messages) ? thread.values.messages : [];
  const firstUserMessage = messages.find((message) => {
    const type = message.type;
    const role = message.role;
    return type === "human" || role === "user";
  });
  const content = typeof firstUserMessage?.content === "string" ? firstUserMessage.content : "";
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) return defaultTitle(thread.thread_id);
  return normalized.length > 56 ? `${normalized.slice(0, 53)}...` : normalized;
}

function sortAndLimit(history: ChatThreadSummary[]): ChatThreadSummary[] {
  const deduped = new Map<string, ChatThreadSummary>();
  for (const thread of history) {
    const previous = deduped.get(thread.id);
    if (!previous || thread.updatedAt >= previous.updatedAt) {
      deduped.set(thread.id, thread);
    }
  }
  return [...deduped.values()]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_STORED_THREADS);
}

function updateThreadHistoryTitle(
  history: ChatThreadSummary[],
  threadId: string,
  title: string,
): ChatThreadSummary[] {
  const now = Date.now();
  const existing = history.find((thread) => thread.id === threadId);
  if (!existing) {
    return sortAndLimit([{ id: threadId, title, createdAt: now, updatedAt: now }, ...history]);
  }
  if (existing.title === title) return history;
  return sortAndLimit(
    history.map((thread) =>
      thread.id === threadId ? { ...thread, title, updatedAt: now } : thread,
    ),
  );
}

function sameThreadHistory(a: ChatThreadSummary[], b: ChatThreadSummary[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((thread, index) => {
    const other = b[index];
    return (
      thread.id === other.id &&
      thread.title === other.title &&
      thread.createdAt === other.createdAt &&
      thread.updatedAt === other.updatedAt
    );
  });
}

export async function loadThreadHistory(
  client: ThreadHistoryClient,
): Promise<ChatThreadSummary[]> {
  const threads = await client.threads.search({
    limit: MAX_STORED_THREADS,
    sortBy: "updated_at",
    sortOrder: "desc",
    select: ["thread_id", "created_at", "updated_at", "values"],
  });
  return threads.map((thread) => {
    const createdAtFallback = Date.now();
    const createdAt = parseTimestamp(thread.created_at, createdAtFallback);
    const updatedAt = parseTimestamp(thread.updated_at, createdAt);
    return {
      id: thread.thread_id,
      title: deriveThreadTitle(thread),
      createdAt,
      updatedAt,
    };
  });
}

export function createInitialState(): SessionState {
  const threadId = readStoredThreadId();
  return {
    threadId,
    threadHistory: [],
    hydrated: true,
  };
}

function emitSessionChange(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(CHAT_SESSION_EVENT));
}

function subscribeToSessionState(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  const handleStorage = (event: StorageEvent) => {
    if (event.key != null && event.key !== THREAD_ID_STORAGE_KEY) {
      return;
    }
    onStoreChange();
  };
  window.addEventListener("storage", handleStorage);
  window.addEventListener(CHAT_SESSION_EVENT, onStoreChange);
  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(CHAT_SESSION_EVENT, onStoreChange);
  };
}

function getSessionSnapshot(): SessionState {
  if (cachedSessionSnapshot == null) {
    cachedSessionSnapshot = createInitialState();
    return cachedSessionSnapshot;
  }
  const storedThreadId = readStoredThreadId();
  if (cachedSessionSnapshot.threadId !== storedThreadId) {
    cachedSessionSnapshot = { ...cachedSessionSnapshot, threadId: storedThreadId };
  }
  return cachedSessionSnapshot;
}

function getServerSessionSnapshot(): SessionState {
  return EMPTY_SESSION_STATE;
}

function writeSessionState(state: SessionState): void {
  if (typeof window === "undefined") return;
  cachedSessionSnapshot = state;
  try {
    if (state.threadId) {
      window.localStorage.setItem(THREAD_ID_STORAGE_KEY, state.threadId);
    } else {
      window.localStorage.removeItem(THREAD_ID_STORAGE_KEY);
    }
  } catch {
    // ignore
  }
  emitSessionChange();
}

export function useChatSession() {
  const state = useSyncExternalStore(
    subscribeToSessionState,
    getSessionSnapshot,
    getServerSessionSnapshot,
  );
  const [sessionId] = useState(() => generateSessionId());

  useEffect(() => {
    emitSessionChange();
  }, []);

  const setThreadId = useCallback((threadId: string | null) => {
    const nextThreadId = threadId?.trim() || null;
    if (state.threadId === nextThreadId) return;
    writeSessionState({ ...state, threadId: nextThreadId, hydrated: true });
  }, [state]);

  const startNewChat = useCallback(() => {
    writeSessionState({ ...state, threadId: null, hydrated: true });
  }, [state]);

  const updateThreadTitle = useCallback((threadId: string, title: string) => {
    const cleanTitle = title.trim();
    if (!threadId.trim() || !cleanTitle) return;
    const threadHistory = updateThreadHistoryTitle(
      state.threadHistory,
      threadId,
      cleanTitle,
    );
    if (threadHistory === state.threadHistory) return;
    writeSessionState({ ...state, threadHistory, hydrated: true });
  }, [state]);

  const refreshThreadHistory = useCallback(async (client: ThreadHistoryClient) => {
    const loaded = await loadThreadHistory(client);
    const snapshot = getSessionSnapshot();
    const threadHistory = sortAndLimit(loaded);
    debugChatStage("refreshThreadHistory", {
      loadedThreadIds: loaded.map((thread) => thread.id),
      existingThreadIds: snapshot.threadHistory.map((thread) => thread.id),
      nextThreadIds: threadHistory.map((thread) => thread.id),
    });
    if (sameThreadHistory(threadHistory, snapshot.threadHistory)) return;
    writeSessionState({ ...snapshot, threadHistory, hydrated: true });
  }, []);

  function clearChat<TMessage, TContext>(helpers: {
    threadId?: string | null;
    setMessages?: (value: TMessage[] | ((prev: TMessage[]) => TMessage[])) => void;
    setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
    setContextUsage: (
      value: TContext | null | ((prev: TContext | null) => TContext | null),
    ) => void;
  }): void {
    const previousThreadId = helpers.threadId === undefined ? state.threadId : helpers.threadId;
    debugChatStage("clearChatSession", {
      previousThreadId,
      removedFromHistory: Boolean(previousThreadId),
    });
    writeSessionState({
      threadId: null,
      threadHistory: previousThreadId
        ? state.threadHistory.filter((thread) => thread.id !== previousThreadId)
        : state.threadHistory,
      hydrated: true,
    });
    helpers.setMessages?.([]);
    helpers.setFeedbackSubmitted(false);
    helpers.setContextUsage(null);
  }

  return {
    threadId: state.threadId,
    setThreadId,
    sessionId,
    clearChat,
    threadHistory: state.threadHistory,
    startNewChat,
    updateThreadTitle,
    refreshThreadHistory,
    isReady: state.hydrated,
  };
}

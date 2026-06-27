import { useCallback, useEffect, useState } from "react";
import { THREAD_ID_STORAGE_KEY } from "@/constants/chat";

const MAX_STORED_THREADS = 30;

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

function parseTimestamp(value: string, fallback: number): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : fallback;
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

function mergeThreadHistory(
  loaded: ChatThreadSummary[],
  existing: ChatThreadSummary[],
): ChatThreadSummary[] {
  const overlayById = new Map(existing.map((thread) => [thread.id, thread]));
  return sortAndLimit(
    loaded.map((thread) => {
      const overlay = overlayById.get(thread.id);
      if (!overlay || overlay.title === defaultTitle(thread.id)) {
        return thread;
      }
      return { ...thread, title: overlay.title };
    }),
  );
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

export function useChatSession() {
  const [state, setState] = useState<SessionState>(EMPTY_SESSION_STATE);
  const [sessionId] = useState(() => generateSessionId());

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setState(createInitialState());
    }, 0);
    return () => window.clearTimeout(timeout);
  }, []);

  useEffect(() => {
    if (!state.threadId) return;
    try {
      window.localStorage.setItem(THREAD_ID_STORAGE_KEY, state.threadId);
    } catch {
      // ignore
    }
  }, [state.threadId]);

  const setThreadId = useCallback((threadId: string | null) => {
    const nextThreadId = threadId?.trim() || null;
    setState((previous) =>
      previous.threadId === nextThreadId ? previous : { ...previous, threadId: nextThreadId },
    );
  }, []);

  const startNewChat = useCallback(() => {
    try {
      window.localStorage.removeItem(THREAD_ID_STORAGE_KEY);
    } catch {
      // ignore
    }
    setState((previous) => ({ ...previous, threadId: null }));
  }, []);

  const updateThreadTitle = useCallback((threadId: string, title: string) => {
    const cleanTitle = title.trim();
    if (!threadId.trim() || !cleanTitle) return;
    setState((previous) => {
      const threadHistory = updateThreadHistoryTitle(
        previous.threadHistory,
        threadId,
        cleanTitle,
      );
      return threadHistory === previous.threadHistory ? previous : { ...previous, threadHistory };
    });
  }, []);

  const refreshThreadHistory = useCallback(async (client: ThreadHistoryClient) => {
    const loaded = await loadThreadHistory(client);
    setState((previous) => {
      const threadHistory = mergeThreadHistory(loaded, previous.threadHistory);
      return sameThreadHistory(threadHistory, previous.threadHistory)
        ? previous
        : { ...previous, threadHistory };
    });
  }, []);

  function clearChat<TMessage, TContext>(helpers: {
    setMessages?: (value: TMessage[] | ((prev: TMessage[]) => TMessage[])) => void;
    setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
    setContextUsage: (
      value: TContext | null | ((prev: TContext | null) => TContext | null),
    ) => void;
  }): void {
    const previousThreadId = state.threadId;
    try {
      window.localStorage.removeItem(THREAD_ID_STORAGE_KEY);
    } catch {
      // ignore
    }
    setState((previous) => ({
      threadId: null,
      threadHistory: previousThreadId
        ? previous.threadHistory.filter((thread) => thread.id !== previousThreadId)
        : previous.threadHistory,
      hydrated: true,
    }));
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

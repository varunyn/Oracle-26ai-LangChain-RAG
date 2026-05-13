import { useCallback, useEffect, useState } from "react";
import {
  CHAT_THREAD_HISTORY_STORAGE_KEY,
  THREAD_ID_STORAGE_KEY,
} from "@/constants/chat";
import { generateThreadId } from "@/lib/chat/messages";

const MAX_STORED_THREADS = 30;

export type ChatThreadSummary = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
};

type SessionState = {
  threadId: string;
  threadHistory: ChatThreadSummary[];
};

const EMPTY_SESSION_STATE: SessionState = {
  threadId: "",
  threadHistory: [],
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

function readStoredThreadHistory(): ChatThreadSummary[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CHAT_THREAD_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item): ChatThreadSummary | null => {
        if (!item || typeof item !== "object") return null;
        const data = item as Record<string, unknown>;
        const id = typeof data.id === "string" ? data.id.trim() : "";
        if (!id) return null;
        const title =
          typeof data.title === "string" && data.title.trim()
            ? data.title.trim()
            : defaultTitle(id);
        const createdAt = typeof data.createdAt === "number" ? data.createdAt : Date.now();
        const updatedAt = typeof data.updatedAt === "number" ? data.updatedAt : createdAt;
        return { id, title, createdAt, updatedAt };
      })
      .filter((item): item is ChatThreadSummary => item != null);
  } catch {
    return [];
  }
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

function createInitialState(): SessionState {
  const threadId = readStoredThreadId() ?? generateThreadId();
  return {
    threadId,
    threadHistory: sortAndLimit(readStoredThreadHistory()),
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
      window.localStorage.setItem(
        CHAT_THREAD_HISTORY_STORAGE_KEY,
        JSON.stringify(state.threadHistory),
      );
    } catch {
      // ignore
    }
  }, [state.threadId, state.threadHistory]);

  const setThreadId = useCallback((threadId: string) => {
    const nextThreadId = threadId.trim();
    if (!nextThreadId) return;
    setState((previous) =>
      previous.threadId === nextThreadId ? previous : { ...previous, threadId: nextThreadId },
    );
  }, []);

  const startNewChat = useCallback(() => {
    const nextThreadId = generateThreadId();
    setState((previous) => ({ ...previous, threadId: nextThreadId }));
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

  function clearChat<TMessage, TContext>(helpers: {
    setMessages?: (value: TMessage[] | ((prev: TMessage[]) => TMessage[])) => void;
    setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
    setContextUsage: (
      value: TContext | null | ((prev: TContext | null) => TContext | null),
    ) => void;
  }): void {
    const previousThreadId = state.threadId;
    const nextThreadId = generateThreadId();
    setState((previous) => ({
      threadId: nextThreadId,
      threadHistory: previous.threadHistory.filter((thread) => thread.id !== previousThreadId),
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
    isReady: state.threadId.length > 0,
  };
}

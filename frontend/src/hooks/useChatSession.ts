import { useEffect, useState } from "react";
import { THREAD_ID_STORAGE_KEY } from "@/constants/chat";
import { generateThreadId } from "@/lib/chat/messages";

/** Session ID: new per tab load/refresh (not persisted). Used for Langfuse session grouping. */
function generateSessionId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID)
    return crypto.randomUUID();
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}

export function useChatSession() {
  const [threadId, setThreadId] = useState(() => {
    if (typeof window !== "undefined") {
      try {
        const storedThreadId = window.localStorage.getItem(THREAD_ID_STORAGE_KEY);
        if (storedThreadId && storedThreadId.trim().length > 0) {
          return storedThreadId;
        }
      } catch {
        // ignore
      }
    }
    return generateThreadId();
  });
  const [sessionId] = useState(() => generateSessionId());

  useEffect(() => {
    try {
      window.localStorage.setItem(THREAD_ID_STORAGE_KEY, threadId);
    } catch {
      // ignore
    }
  }, [threadId]);

  function clearChat<TMessage, TContext>(helpers: {
    setMessages?: (value: TMessage[] | ((prev: TMessage[]) => TMessage[])) => void;
    setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
    setContextUsage: (
      value: TContext | null | ((prev: TContext | null) => TContext | null),
    ) => void;
  }): void {
    const nextThreadId = generateThreadId();
    setThreadId(nextThreadId);
    helpers.setMessages?.([]);
    helpers.setFeedbackSubmitted(false);
    helpers.setContextUsage(null);
  }

  return { threadId, setThreadId, sessionId, clearChat };
}

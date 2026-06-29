"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useStream } from "@langchain/react";
import { debugChatStage } from "@/hooks/chat/debug";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";
import type { BaseMessageWithKwargs } from "@/hooks/chat/references";

type StreamValue = ReturnType<typeof useStream>;

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function threadMessagesFromStatePayload(payload: unknown): BaseMessageWithKwargs[] | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const values = (payload as { values?: { messages?: unknown } }).values;
  return Array.isArray(values?.messages) ? (values.messages as BaseMessageWithKwargs[]) : undefined;
}

function threadMessagesFromSearchPayload(
  payload: unknown,
  threadId: string | null,
): BaseMessageWithKwargs[] | undefined {
  if (!threadId || !Array.isArray(payload)) return undefined;
  const match = payload.find((thread) => {
    return (
      thread != null &&
      typeof thread === "object" &&
      (thread as { thread_id?: unknown }).thread_id === threadId
    );
  }) as { values?: { messages?: unknown } } | undefined;
  return Array.isArray(match?.values?.messages)
    ? (match.values.messages as BaseMessageWithKwargs[])
    : undefined;
}

type LangGraphStreamContextValue = {
  threadId: string | null;
  setThreadId: Dispatch<SetStateAction<string | null>> | ((threadId: string | null) => void);
  serverThreadMessages: BaseMessageWithKwargs[] | undefined;
  stream: StreamValue;
  transportError: Error | null;
};

const LangGraphStreamContext = createContext<LangGraphStreamContextValue | null>(null);

export function LangGraphStreamProvider({
  threadId,
  setThreadId,
  children,
}: {
  threadId: string | null;
  setThreadId: Dispatch<SetStateAction<string | null>> | ((threadId: string | null) => void);
  children: ReactNode;
}) {
  const langgraphApiUrl = useMemo(() => resolveLanggraphApiUrl(), []);
  const [transportErrorState, setTransportErrorState] = useState<{
    error: Error | null;
    threadId: string | null;
  }>({
    error: null,
    threadId,
  });
  const [serverThreadMessagesState, setServerThreadMessagesState] = useState<{
    messages: BaseMessageWithKwargs[] | undefined;
    threadId: string | null;
  }>({
    messages: undefined,
    threadId,
  });
  const threadIdRef = useRef<string | null>(threadId);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    threadIdRef.current = threadId;
    return () => {
      mountedRef.current = false;
    };
  }, [threadId]);

  const transportError =
    transportErrorState.threadId === threadId ? transportErrorState.error : null;
  const serverThreadMessages =
    serverThreadMessagesState.threadId === threadId
      ? serverThreadMessagesState.messages
      : undefined;

  const instrumentedFetch = useMemo(() => {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const requestUrl =
        typeof input === "string"
          ? input
          : input instanceof Request
            ? input.url
            : input.toString();
      try {
        const response = await fetch(input, init);
        const nextError =
          response.ok ? null : new Error(`Protocol request failed: ${response.status} ${response.statusText}`);
        if (
          mountedRef.current &&
          (requestUrl.includes("/threads/") || requestUrl.includes("/threads/search"))
        ) {
          setTransportErrorState({ error: nextError, threadId: threadIdRef.current });
        }
        if (response.ok && requestUrl.endsWith("/state")) {
          const payload = await response.clone().json().catch(() => null);
          const messages = threadMessagesFromStatePayload(payload);
          if (messages && mountedRef.current) {
            setServerThreadMessagesState({ messages, threadId: threadIdRef.current });
          }
        }
        if (response.ok && requestUrl.includes("/threads/search")) {
          const payload = await response.clone().json().catch(() => null);
          const messages = threadMessagesFromSearchPayload(payload, threadIdRef.current);
          if (messages && mountedRef.current) {
            setServerThreadMessagesState({ messages, threadId: threadIdRef.current });
          }
        }
        return response;
      } catch (error) {
        if (
          mountedRef.current &&
          (requestUrl.includes("/threads/") || requestUrl.includes("/threads/search"))
        ) {
          setTransportErrorState({
            error: toError(error),
            threadId: threadIdRef.current,
          });
        }
        throw error;
      }
    };
  }, []);

  const stream = useStream({
    apiUrl: langgraphApiUrl,
    assistantId: "chat_agent",
    fetch: instrumentedFetch,
    threadId,
    onThreadId: (id) => {
      if (id) {
        debugChatStage("LangGraphStreamProvider.onThreadId", {
          previousThreadId: threadId,
          nextThreadId: id,
        });
        setThreadId(id);
      }
    },
  });

  useEffect(() => {
    debugChatStage("LangGraphStreamProvider.state", {
      threadId,
      streamThreadId: stream.threadId,
      messageCount: Array.isArray(stream.messages) ? stream.messages.length : 0,
      toolCallCount: Array.isArray(stream.toolCalls) ? stream.toolCalls.length : 0,
      isLoading: stream.isLoading,
      hasError: stream.error != null,
    });
  }, [stream.error, stream.isLoading, stream.messages, stream.threadId, stream.toolCalls, threadId]);

  return (
    <LangGraphStreamContext.Provider
      value={{ threadId, setThreadId, serverThreadMessages, stream, transportError }}
    >
      {children}
    </LangGraphStreamContext.Provider>
  );
}

export function useLangGraphStream(): LangGraphStreamContextValue {
  const value = useContext(LangGraphStreamContext);
  if (value == null) {
    throw new Error("useLangGraphStream must be used within LangGraphStreamProvider");
  }
  return value;
}

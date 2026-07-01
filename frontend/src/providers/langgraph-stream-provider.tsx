"use client";

import { useStream } from "@langchain/react";
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { debugChatStage } from "@/hooks/chat/debug";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";
import type { BaseMessageWithKwargs } from "@/hooks/chat/references";

type StreamValue = ReturnType<typeof useStream>;
type RunCreatedInfo = { runId: string };
type RunCompletedInfo = {
  runId?: string;
  reason: "success" | "error" | "interrupt" | "stopped";
};

function threadMessagesFromStatePayload(
  payload: unknown
): BaseMessageWithKwargs[] | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const values = (payload as { values?: { messages?: unknown } }).values;
  return Array.isArray(values?.messages)
    ? (values.messages as BaseMessageWithKwargs[])
    : undefined;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export function formatProtocolRequestError(
  status: number,
  statusText: string,
  requestUrl: string
): string {
  return `Protocol request failed: ${status} ${statusText} (${requestUrl})`;
}

type LangGraphStreamContextValue = {
  threadId: string | null;
  setThreadId:
    | Dispatch<SetStateAction<string | null>>
    | ((threadId: string | null) => void);
  authoritativeThreadMessages: BaseMessageWithKwargs[] | undefined;
  stream: StreamValue;
  transportError: Error | null;
};

const LangGraphStreamContext =
  createContext<LangGraphStreamContextValue | null>(null);

export function LangGraphStreamProvider({
  threadId,
  setThreadId,
  children,
}: {
  threadId: string | null;
  setThreadId:
    | Dispatch<SetStateAction<string | null>>
    | ((threadId: string | null) => void);
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
  const [authoritativeThreadMessagesState, setAuthoritativeThreadMessagesState] =
    useState<{
      messages: BaseMessageWithKwargs[] | undefined;
      threadId: string | null;
    }>({ messages: undefined, threadId });
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
    transportErrorState.threadId === threadId
      ? transportErrorState.error
      : null;
  const authoritativeThreadMessages =
    authoritativeThreadMessagesState.threadId === threadId
      ? authoritativeThreadMessagesState.messages
      : undefined;

  const instrumentedFetch = useMemo(
    () =>
      async (
        input: RequestInfo | URL,
        init?: RequestInit
      ): Promise<Response> => {
        const requestUrl =
          typeof input === "string"
            ? input
            : input instanceof Request
              ? input.url
              : input.toString();
        try {
          const response = await fetch(input, init);
          const nextError = response.ok
            ? null
            : new Error(
                formatProtocolRequestError(
                  response.status,
                  response.statusText,
                  requestUrl
                )
              );
          if (
            mountedRef.current &&
            (requestUrl.includes("/threads/") ||
              requestUrl.includes("/threads/search"))
          ) {
            setTransportErrorState({
              error: nextError,
              threadId: threadIdRef.current,
            });
          }
          return response;
        } catch (error) {
          if (
            mountedRef.current &&
            (requestUrl.includes("/threads/") ||
              requestUrl.includes("/threads/search"))
          ) {
            setTransportErrorState({
              error: toError(error),
              threadId: threadIdRef.current,
            });
          }
          throw error;
        }
      },
    []
  );

  const handleCreated = useCallback((info: RunCreatedInfo) => {
    debugChatStage("LangGraphStreamProvider.onCreated", { runId: info.runId });
  }, []);

  const hydrateAuthoritativeThread = useCallback(
    async (completedThreadId: string) => {
      const response = await instrumentedFetch(
        `${langgraphApiUrl}/threads/${completedThreadId}/state`
      );
      const payload = await response.json();
      const messages = threadMessagesFromStatePayload(payload);
      if (!messages || !mountedRef.current) return;

      setAuthoritativeThreadMessagesState({
        messages,
        threadId: completedThreadId,
      });
      debugChatStage("LangGraphStreamProvider.authoritativeState", {
        threadId: completedThreadId,
        messageCount: messages.length,
        messageIds: messages.map((message) => message.id),
      });
    },
    [instrumentedFetch, langgraphApiUrl]
  );

  const handleCompleted = useCallback(
    (info: RunCompletedInfo) => {
      const completedThreadId = threadIdRef.current;
      debugChatStage("LangGraphStreamProvider.onCompleted", {
        threadId: completedThreadId,
        reason: info.reason,
        runId: info.runId,
      });
      if (info.reason !== "success" || !completedThreadId) return;

      void hydrateAuthoritativeThread(completedThreadId).catch((error) => {
        debugChatStage("LangGraphStreamProvider.authoritativeStateError", {
          threadId: completedThreadId,
          error: String(error),
        });
      });
    },
    [hydrateAuthoritativeThread]
  );

  const stream = useStream({
    apiUrl: langgraphApiUrl,
    assistantId: "chat_agent",
    fetch: instrumentedFetch,
    threadId,
    onCreated: handleCreated,
    onCompleted: handleCompleted,
    onThreadId: (id) => {
      if (id) {
        threadIdRef.current = id;
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
      toolCallCount: Array.isArray(stream.toolCalls)
        ? stream.toolCalls.length
        : 0,
      isLoading: stream.isLoading,
      hasError: stream.error != null,
    });
  }, [
    stream.error,
    stream.isLoading,
    stream.messages,
    stream.threadId,
    stream.toolCalls,
    threadId,
  ]);

  return (
    <LangGraphStreamContext.Provider
      value={{
        threadId,
        setThreadId,
        authoritativeThreadMessages,
        stream,
        transportError,
      }}
    >
      {children}
    </LangGraphStreamContext.Provider>
  );
}

export function useLangGraphStream(): LangGraphStreamContextValue {
  const value = useContext(LangGraphStreamContext);
  if (value == null) {
    throw new Error(
      "useLangGraphStream must be used within LangGraphStreamProvider"
    );
  }
  return value;
}

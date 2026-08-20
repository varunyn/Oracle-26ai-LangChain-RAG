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

type StreamValue = ReturnType<typeof useStream>;
type RunCreatedInfo = { runId: string };

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

  const stream = useStream({
    apiUrl: langgraphApiUrl,
    assistantId: "chat_agent",
    fetch: instrumentedFetch,
    threadId,
    onCreated: handleCreated,
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

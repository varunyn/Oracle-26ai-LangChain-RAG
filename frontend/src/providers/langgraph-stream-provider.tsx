"use client";

import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useChannel, useStream } from "@langchain/react";
import { debugChatStage } from "@/hooks/chat/debug";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";
import type { BaseMessageWithKwargs } from "@/hooks/chat/references";
import {
  projectMcpToolActivities,
  type McpToolActivity,
} from "@/lib/types/mcp-activity";

type StreamValue = ReturnType<typeof useStream>;
type RunCompletedInfo = {
  runId?: string;
  reason: "success" | "error" | "interrupt" | "stopped";
};
type RunCreatedInfo = { runId: string };

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function threadMessagesFromStatePayload(payload: unknown): BaseMessageWithKwargs[] | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const values = (payload as { values?: { messages?: unknown } }).values;
  return Array.isArray(values?.messages) ? (values.messages as BaseMessageWithKwargs[]) : undefined;
}

type LangGraphStreamContextValue = {
  threadId: string | null;
  setThreadId: Dispatch<SetStateAction<string | null>> | ((threadId: string | null) => void);
  authoritativeThreadMessages: BaseMessageWithKwargs[] | undefined;
  mcpToolActivities: McpToolActivity[];
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
  const [authoritativeThreadMessagesState, setAuthoritativeThreadMessagesState] = useState<{
    messages: BaseMessageWithKwargs[] | undefined;
    threadId: string | null;
  }>({
    messages: undefined,
    threadId,
  });
  const threadIdRef = useRef<string | null>(threadId);
  const mcpActivityEventsRef = useRef<readonly unknown[]>([]);
  const [mcpActivityStartIndex, setMcpActivityStartIndex] = useState(0);
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
  const authoritativeThreadMessages =
    authoritativeThreadMessagesState.threadId === threadId
      ? authoritativeThreadMessagesState.messages
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

  const hydrateAuthoritativeThread = useCallback(
    async (completedThreadId: string) => {
      const response = await instrumentedFetch(
        `${langgraphApiUrl}/threads/${completedThreadId}/state`,
      );
      const payload = await response.clone().json().catch(() => null);
      const messages = threadMessagesFromStatePayload(payload);
      if (messages && mountedRef.current) {
        setAuthoritativeThreadMessagesState({
          messages,
          threadId: completedThreadId,
        });
        debugChatStage("LangGraphStreamProvider.authoritativeState", {
          threadId: completedThreadId,
          messageCount: messages.length,
        });
      }
    },
    [instrumentedFetch, langgraphApiUrl],
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
    [hydrateAuthoritativeThread],
  );

  const handleCreated = useCallback((info: RunCreatedInfo) => {
    setMcpActivityStartIndex(mcpActivityEventsRef.current.length);
    debugChatStage("LangGraphStreamProvider.onCreated", { runId: info.runId });
  }, []);

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

  const mcpActivityEvents = useChannel(stream, ["custom:mcp_tool_activity"]);
  useEffect(() => {
    mcpActivityEventsRef.current = mcpActivityEvents;
  }, [mcpActivityEvents]);
  const mcpToolActivities = useMemo(
    () => projectMcpToolActivities(mcpActivityEvents.slice(mcpActivityStartIndex)),
    [mcpActivityEvents, mcpActivityStartIndex],
  );

  useEffect(() => {
    debugChatStage("LangGraphStreamProvider.mcpActivity", {
      eventCount: mcpActivityEvents.length,
      startIndex: mcpActivityStartIndex,
      activityCount: mcpToolActivities.length,
      events: mcpActivityEvents.slice(-3),
      activities: mcpToolActivities.map(({ toolRunId, toolName, status }) => ({
        toolRunId,
        toolName,
        status,
      })),
    });
  }, [mcpActivityEvents, mcpActivityStartIndex, mcpToolActivities]);

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
      value={{
        threadId,
        setThreadId,
        authoritativeThreadMessages,
        mcpToolActivities,
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
    throw new Error("useLangGraphStream must be used within LangGraphStreamProvider");
  }
  return value;
}

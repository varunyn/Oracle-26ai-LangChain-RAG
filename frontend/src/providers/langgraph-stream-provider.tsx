"use client";

import {
  StreamProvider,
  type UseStreamReturn,
  useStreamContext,
} from "@langchain/react";
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { debugChatStage } from "@/hooks/chat/debug";
import type { PublicChatGraphState } from "@/hooks/chat/graph-state";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";

type ChatStream = UseStreamReturn<PublicChatGraphState>;

export interface ReconnectState {
  attempt: number;
  cause: unknown;
  delayMs: number;
}

interface LangGraphStreamContextValue {
  reconnect: ReconnectState | null;
  setThreadId:
    | Dispatch<SetStateAction<string | null>>
    | ((threadId: string | null) => void);
  stream: ChatStream;
  threadId: string | null;
}

const LangGraphStreamContext =
  createContext<LangGraphStreamContextValue | null>(null);

function StreamBridge({
  threadId,
  setThreadId,
  children,
  reconnect,
}: {
  threadId: string | null;
  setThreadId:
    | Dispatch<SetStateAction<string | null>>
    | ((threadId: string | null) => void);
  children: ReactNode;
  reconnect: ReconnectState | null;
}) {
  const stream = useStreamContext<PublicChatGraphState>();

  useEffect(() => {
    debugChatStage("LangGraphStreamProvider.state", {
      threadId,
      streamThreadId: stream.threadId,
      messageCount: stream.messages.length,
      toolCallCount: stream.toolCalls.length,
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
      value={{ threadId, setThreadId, stream, reconnect }}
    >
      {children}
    </LangGraphStreamContext.Provider>
  );
}

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
  const [reconnect, setReconnect] = useState<ReconnectState | null>(null);
  const mountedRef = useRef(false);
  const pendingReconnectRef = useRef<ReconnectState | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    if (pendingReconnectRef.current != null) {
      setReconnect(pendingReconnectRef.current);
      pendingReconnectRef.current = null;
    }
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const handleThreadId = useCallback(
    (nextThreadId: string) => setThreadId(nextThreadId),
    [setThreadId]
  );

  const handleReconnect = useCallback((next: ReconnectState) => {
    if (!mountedRef.current) {
      pendingReconnectRef.current = next;
      return;
    }
    setReconnect(next);
  }, []);
  const handleConnected = useCallback(() => {
    pendingReconnectRef.current = null;
    if (!mountedRef.current) {
      return;
    }
    setReconnect(null);
  }, []);

  return (
    <StreamProvider<PublicChatGraphState>
      apiUrl={resolveLanggraphApiUrl()}
      assistantId="chat_agent"
      maxReconnectAttempts={3}
      onConnected={handleConnected}
      onReconnect={handleReconnect}
      onThreadId={handleThreadId}
      threadId={threadId}
    >
      <StreamBridge
        reconnect={reconnect}
        setThreadId={setThreadId}
        threadId={threadId}
      >
        {children}
      </StreamBridge>
    </StreamProvider>
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

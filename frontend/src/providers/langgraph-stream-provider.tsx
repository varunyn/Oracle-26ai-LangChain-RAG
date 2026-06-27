"use client";

import {
  createContext,
  useContext,
  useMemo,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useStream } from "@langchain/react";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";

type StreamValue = ReturnType<typeof useStream>;

type LangGraphStreamContextValue = {
  threadId: string | null;
  setThreadId: Dispatch<SetStateAction<string | null>> | ((threadId: string | null) => void);
  stream: StreamValue;
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
  const stream = useStream({
    apiUrl: langgraphApiUrl,
    assistantId: "chat_agent",
    threadId,
    onThreadId: (id) => {
      if (id) {
        setThreadId(id);
      }
    },
  });

  return (
    <LangGraphStreamContext.Provider value={{ threadId, setThreadId, stream }}>
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

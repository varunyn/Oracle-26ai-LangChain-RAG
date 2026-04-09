import { useMemo } from "react";

export type FlowMode = "rag" | "mcp" | "mixed" | "direct";

type ChatBodyParams = {
  model: string;
  thread_id?: string;
  session_id?: string;
  collection_name?: string;
  enable_reranker: boolean;
  enable_tracing: boolean;
  mode: FlowMode;
};

export function useChatBodyParams({
  selectedModel,
  threadId,
  sessionId,
  collectionName,
  enableReranker,
  enableTracing,
  flowMode,
}: {
  selectedModel: string;
  threadId: string;
  sessionId: string;
  collectionName: string;
  enableReranker: boolean;
  enableTracing: boolean;
  flowMode: FlowMode;
}): ChatBodyParams {
  return useMemo(
    () => ({
      model: selectedModel,
      thread_id: threadId || undefined,
      session_id: sessionId || undefined,
      collection_name: collectionName || undefined,
      enable_reranker: enableReranker,
      enable_tracing: enableTracing,
      mode: flowMode,
    }),
    [
      selectedModel,
      threadId,
      sessionId,
      collectionName,
      enableReranker,
      enableTracing,
      flowMode,
    ],
  );
}

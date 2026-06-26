import type { FlowMode } from "@/hooks/useChatBodyParams";
import { getClientApiBase } from "@/lib/api-base";

export type ChatBodyParams = {
  model: string;
  thread_id?: string;
  session_id?: string;
  collection_name?: string;
  enable_reranker: boolean;
  enable_tracing: boolean;
  mode: FlowMode;
};

export function resolveLanggraphApiUrl(): string {
  const base = getClientApiBase();
  if (!base && typeof window !== "undefined") {
    return `${window.location.origin}/api/langgraph`;
  }
  return `${base}/api/langgraph`;
}

export function buildSubmitPayload(
  text: string,
  bodyParams: ChatBodyParams,
  mode: FlowMode,
  messageId?: string,
) {
  return {
    messages: [{ id: messageId, type: "human", content: text }],
    model: bodyParams.model,
    session_id: bodyParams.session_id,
    collection_name: bodyParams.collection_name,
    enable_reranker: bodyParams.enable_reranker,
    enable_tracing: bodyParams.enable_tracing,
    mode,
    context: { ...bodyParams, mode },
    metadata: { ...bodyParams, mode },
    configurable: { ...bodyParams, mode },
  };
}

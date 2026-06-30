import type { FlowMode } from "@/hooks/useChatBodyParams";

export type ChatBodyParams = {
  model: string;
  thread_id?: string;
  session_id?: string;
  collection_name?: string;
  enable_reranker: boolean;
  enable_tracing: boolean;
  mode: FlowMode;
};

function normalizeBase(base: string): string {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

export function resolveLanggraphApiUrl(): string {
  const configuredBase = process.env.NEXT_PUBLIC_LANGGRAPH_API_BASE?.trim();
  if (configuredBase) {
    return normalizeBase(configuredBase);
  }
  if (typeof window !== "undefined") {
    return "http://localhost:2024";
  }
  return normalizeBase(
    process.env.LANGGRAPH_BACKEND_URL || "http://localhost:2024"
  );
}

export function buildLangGraphSubmitPayload(
  text: string,
  bodyParams: ChatBodyParams
) {
  const message: {
    type: "human";
    content: string;
  } = {
    type: "human",
    content: text,
  };

  const configurable: Record<string, unknown> = {
    model_id: bodyParams.model,
    mode: bodyParams.mode,
    enable_reranker: bodyParams.enable_reranker,
    enable_tracing: bodyParams.enable_tracing,
  };
  if (bodyParams.collection_name) {
    configurable.collection_name = bodyParams.collection_name;
  }
  if (bodyParams.session_id) {
    configurable.session_id = bodyParams.session_id;
  }

  return {
    input: {
      messages: [message],
    },
    config: {
      configurable,
    },
  };
}

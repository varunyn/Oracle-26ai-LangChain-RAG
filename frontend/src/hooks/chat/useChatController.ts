import { AIMessage, HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { useStream } from "@langchain/react";
import { startTransition, useCallback, useEffect, useMemo, useState } from "react";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useChatBodyParams, type FlowMode } from "@/hooks/useChatBodyParams";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { getClientApiBase, toApiUrl } from "@/lib/api-base";
import { getMessageContent } from "@/lib/chat/messages";
import type { McpToolInvocation } from "@/lib/types/chat";

type ContextUsage = {
  tokens: number;
  max: number;
  percent: number;
  model_id?: string;
};

type ToastApi = {
  error: (description: string, title?: string) => void;
  success: (description: string, title?: string) => void;
};

type ReferencePayload = {
  standalone_question?: string;
  citations?: { source: string; page: string }[];
  reranker_docs?: {
    page_content: string;
    metadata: Record<string, unknown>;
  }[];
  context_usage?: ContextUsage;
  mcp_used?: boolean;
  mcp_tools_used?: string[];
  mcp_tool_invocations?: McpToolInvocation[];
  error?: string;
};

type MessageLike = {
  id?: string;
  role?: string;
  content?: string;
  parts?: { type?: string; text?: string; data?: unknown }[];
};
type ChatStatus = "submitted" | "streaming" | "ready" | "error";
type SendOverrides = {
  mode?: FlowMode;
};

type ClearSessionChat = (helpers: {
  setMessages?: (value: MessageLike[] | ((prev: MessageLike[]) => MessageLike[])) => void;
  setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
  setContextUsage: (value: ContextUsage | null | ((prev: ContextUsage | null) => ContextUsage | null)) => void;
}) => void;

type UseChatControllerArgs = {
  selectedModel: string;
  threadId: string;
  sessionId: string;
  collectionName: string;
  enableReranker: boolean;
  enableTracing: boolean;
  flowMode: FlowMode;
  toast: ToastApi;
  clearSessionChat: ClearSessionChat;
};

type BaseMessageWithKwargs = BaseMessage & {
  additional_kwargs?: Record<string, unknown>;
  response_metadata?: Record<string, unknown>;
};

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

function normalizeContextUsage(raw: unknown): ContextUsage | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const usage = raw as Record<string, unknown>;
  const tokens = toFiniteNumber(usage.tokens);
  const max = toFiniteNumber(usage.max);
  const percent = toFiniteNumber(usage.percent);
  if (tokens == null || max == null || percent == null) return undefined;
  return {
    tokens,
    max,
    percent,
    model_id: typeof usage.model_id === "string" ? usage.model_id : undefined,
  };
}

function resolveLanggraphApiUrl(): string {
  const base = getClientApiBase();
  return `${base}/api/langgraph`;
}

function readText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      const text = (part as { text?: unknown }).text;
      return typeof text === "string" ? text : "";
    })
    .join("");
}

function toRole(message: BaseMessage): "user" | "assistant" | "system" {
  if (HumanMessage.isInstance(message)) return "user";
  if (AIMessage.isInstance(message)) return "assistant";
  if (SystemMessage.isInstance(message)) return "system";
  const msgType = (message as { type?: unknown }).type;
  if (msgType === "human") return "user";
  if (msgType === "ai") return "assistant";
  return "system";
}

function toReferences(message: BaseMessageWithKwargs): ReferencePayload | null {
  const candidates: unknown[] = [
    message.additional_kwargs,
    message.additional_kwargs?.references,
    message.response_metadata,
  ];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") continue;
    const raw = candidate as Record<string, unknown>;
    const hasKnownReferenceField =
      Array.isArray(raw.citations) ||
      Array.isArray(raw.reranker_docs) ||
      Array.isArray(raw.mcp_tools_used) ||
      Array.isArray(raw.mcp_tool_invocations) ||
      typeof raw.standalone_question === "string" ||
      typeof raw.error === "string" ||
      raw.mcp_used === true;
    if (!hasKnownReferenceField) continue;
    return {
      standalone_question:
        typeof raw.standalone_question === "string" ? raw.standalone_question : undefined,
      citations: Array.isArray(raw.citations)
        ? (raw.citations as { source: string; page: string }[])
        : [],
      reranker_docs: Array.isArray(raw.reranker_docs)
        ? (raw.reranker_docs as { page_content: string; metadata: Record<string, unknown> }[])
        : [],
      context_usage: normalizeContextUsage(raw.context_usage),
      mcp_used: raw.mcp_used === true,
      mcp_tools_used: Array.isArray(raw.mcp_tools_used)
        ? (raw.mcp_tools_used as string[])
        : undefined,
      mcp_tool_invocations: Array.isArray(raw.mcp_tool_invocations)
        ? (raw.mcp_tool_invocations as McpToolInvocation[])
        : undefined,
      error: typeof raw.error === "string" ? raw.error : undefined,
    };
  }
  return null;
}

function toReferencesFromRawMessage(rawMessage: unknown): ReferencePayload | null {
  if (!rawMessage || typeof rawMessage !== "object") return null;
  const data = rawMessage as Record<string, unknown>;
  const candidates: unknown[] = [
    data.additional_kwargs,
    (data.additional_kwargs as Record<string, unknown> | undefined)?.references,
    data.response_metadata,
    data,
  ];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") continue;
    const raw = candidate as Record<string, unknown>;
    const hasKnownReferenceField =
      Array.isArray(raw.citations) ||
      Array.isArray(raw.reranker_docs) ||
      Array.isArray(raw.mcp_tools_used) ||
      Array.isArray(raw.mcp_tool_invocations) ||
      typeof raw.standalone_question === "string" ||
      typeof raw.error === "string" ||
      raw.mcp_used === true;
    if (!hasKnownReferenceField) continue;
    return {
      standalone_question:
        typeof raw.standalone_question === "string" ? raw.standalone_question : undefined,
      citations: Array.isArray(raw.citations)
        ? (raw.citations as { source: string; page: string }[])
        : [],
      reranker_docs: Array.isArray(raw.reranker_docs)
        ? (raw.reranker_docs as { page_content: string; metadata: Record<string, unknown> }[])
        : [],
      context_usage: normalizeContextUsage(raw.context_usage),
      mcp_used: raw.mcp_used === true,
      mcp_tools_used: Array.isArray(raw.mcp_tools_used)
        ? (raw.mcp_tools_used as string[])
        : undefined,
      mcp_tool_invocations: Array.isArray(raw.mcp_tool_invocations)
        ? (raw.mcp_tool_invocations as McpToolInvocation[])
        : undefined,
      error: typeof raw.error === "string" ? raw.error : undefined,
    };
  }
  return null;
}

function normalizeStatus(rawStatus: unknown, isLoading: boolean, hasError: boolean): ChatStatus {
  if (hasError) return "error";
  if (
    rawStatus === "submitted" ||
    rawStatus === "streaming" ||
    rawStatus === "ready" ||
    rawStatus === "error"
  ) {
    return rawStatus;
  }
  return isLoading ? "streaming" : "ready";
}

function getLastUserMessageText(messages: MessageLike[]): string {
  const lastUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  if (lastUserMessage == null) return "";
  return getMessageContent(lastUserMessage).trim();
}

export function useChatController({
  selectedModel,
  threadId,
  sessionId,
  collectionName,
  enableReranker,
  enableTracing,
  flowMode,
  toast,
  clearSessionChat,
}: UseChatControllerArgs) {
  const [input, setInput] = useState("");
  const [maxCitationsToShow, setMaxCitationsToShow] = useState(10);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);

  const bodyParams = useChatBodyParams({
    selectedModel,
    threadId,
    sessionId,
    collectionName,
    enableReranker,
    enableTracing,
    flowMode,
  });

  const langgraphApiUrl = useMemo(() => resolveLanggraphApiUrl(), []);

  const stream = useStream({
    apiUrl: langgraphApiUrl,
    assistantId: "mcp_agent_executor",
    threadId,
    reconnectOnMount: true,
    fetchStateHistory: true,
    onStop: () => {
      toast.success("Generation stopped");
    },
    onError: (error) => {
      console.error("Chat error:", error);
      const message = error instanceof Error ? error.message : String(error);
      toast.error(message);
    },
  });

  const streamMessages = stream.messages;
  const streamValues = (stream as { values?: unknown }).values;

  const sendUserMessage = useCallback(
    (text: string, overrides?: SendOverrides) => {
      const effectiveMode = overrides?.mode ?? bodyParams.mode;
      stream.submit({
        messages: [{ type: "human", content: text }],
        model: bodyParams.model,
        session_id: bodyParams.session_id,
        collection_name: bodyParams.collection_name,
        enable_reranker: bodyParams.enable_reranker,
        enable_tracing: bodyParams.enable_tracing,
        mode: effectiveMode,
        context: { ...bodyParams, mode: effectiveMode },
        metadata: { ...bodyParams, mode: effectiveMode },
        configurable: { ...bodyParams, mode: effectiveMode },
      });
    },
    [bodyParams, stream],
  );

  const messages = useMemo<MessageLike[]>(() => {
    const raw = (streamMessages ?? []) as BaseMessageWithKwargs[];
    const mapped = raw.map((message, index) => {
      const text = readText(message.content);
      const refData = toReferences(message);
      const parts: { type?: string; text?: string; data?: unknown }[] = [];
      if (text) parts.push({ type: "text", text });
      if (refData) parts.push({ type: "data-references", data: refData });
      return {
        id: typeof message.id === "string" ? message.id : `message-${index}`,
        role: toRole(message),
        content: text,
        parts,
      };
    });

    // Fallback for cases where class-message metadata drops reference payloads.
    const rawValues = streamValues;
    const valueMessages =
      rawValues && typeof rawValues === "object" && Array.isArray((rawValues as { messages?: unknown }).messages)
        ? ((rawValues as { messages: unknown[] }).messages as unknown[])
        : [];
    const lastAssistantValue = [...valueMessages]
      .reverse()
      .find((msg) => {
        if (!msg || typeof msg !== "object") return false;
        const data = msg as Record<string, unknown>;
        const role = String(data.role ?? "").toLowerCase();
        const type = String(data.type ?? "").toLowerCase();
        return role === "assistant" || type === "ai";
      });
    const fallbackRefs = toReferencesFromRawMessage(lastAssistantValue);
    if (!fallbackRefs) return mapped;

    const lastAssistantIdx = [...mapped]
      .map((msg, idx) => ({ msg, idx }))
      .reverse()
      .find(({ msg }) => msg.role === "assistant")?.idx;
    if (lastAssistantIdx == null) return mapped;
    const target = mapped[lastAssistantIdx];
    const hasRefsAlready =
      Array.isArray(target.parts) &&
      target.parts.some((part) => part?.type === "data-references" && part.data != null);
    if (hasRefsAlready) return mapped;

    const next = [...mapped];
    next[lastAssistantIdx] = {
      ...target,
      parts: [...(target.parts ?? []), { type: "data-references", data: fallbackRefs }],
    };
    return next;
  }, [streamMessages, streamValues]);

  useEffect(() => {
    const lastAssistant = [...messages]
      .reverse()
      .find((msg) => msg.role === "assistant");
    if (!lastAssistant?.parts) return;
    const refPart = lastAssistant.parts.find((part) => part.type === "data-references");
    if (!refPart?.data || typeof refPart.data !== "object") return;
    const refs = refPart.data as ReferencePayload;
    const contextUsagePayload = refs.context_usage;
    if (contextUsagePayload) {
      startTransition(() => {
        setContextUsage(contextUsagePayload);
      });
    }
    if (typeof refs.error === "string" && refs.error.length > 0) {
      toast.error(refs.error, "Search unavailable");
    }
  }, [messages, toast]);

  const status = normalizeStatus(
    (stream as { status?: unknown }).status,
    stream.isLoading,
    stream.error != null,
  );

  const {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    showOptimisticSuggestion,
  } = useSuggestions({
    messages,
    status,
    sendMessage: (text) => sendUserMessage(text),
    selectedModel,
    setFeedbackSubmitted,
  });

  const chatContainerRef = useScrollToBottom(status, messages);

  const handleSubmit = useCallback(
    (e: React.SyntheticEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!input.trim()) return;

      setFeedbackSubmitted(false);
      sendUserMessage(input);
      setInput("");
      setMaxCitationsToShow(10);
    },
    [input, sendUserMessage],
  );

  const handleRetry = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;

    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage]);

  const handleRecoverDirect = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "direct" });
  }, [messages, sendUserMessage]);

  const handleRecoverRagOnly = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "rag" });
  }, [messages, sendUserMessage]);

  const handleResumeTurn = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage]);

  const handleFeedback = useCallback(
    async (stars: number) => {
      if (messages.length < 2) return;
      const lastUser = messages[messages.length - 2] as MessageLike | undefined;
      const lastAssistant = messages[messages.length - 1] as MessageLike | undefined;
      if (lastUser?.role !== "user" || lastAssistant?.role !== "assistant") return;

      const question = getMessageContent(lastUser);
      const answer = getMessageContent(lastAssistant);

      try {
        const res = await fetch(toApiUrl("/api/feedback"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, answer, feedback: stars }),
        });
        if (res.ok) setFeedbackSubmitted(true);
      } catch (error) {
        console.error("Feedback submission failed:", error);
        toast.error("Failed to submit feedback");
      }
    },
    [messages, toast],
  );

  const handleClearChat = useCallback(async () => {
    try {
      await fetch(toApiUrl(`/api/threads/${encodeURIComponent(threadId)}`), {
        method: "DELETE",
      });
    } catch (error) {
      console.error("Thread cleanup failed:", error);
    }

    if (typeof stream.stop === "function") {
      stream.stop();
    }

    clearSessionChat({
      setFeedbackSubmitted,
      setContextUsage,
    });
    toast.success("Chat cleared");
  }, [clearSessionChat, threadId, toast, stream]);

  const canStopStream = status === "submitted" || status === "streaming";
  const canResumeTurn = status === "error" && getLastUserMessageText(messages).length > 0;

  return {
    input,
    setInput,
    messages,
    status,
    maxCitationsToShow,
    chatContainerRef,
    handleSubmit,
    canStopStream,
    canResumeTurn,
    handleResumeTurn,
    handleRecoverDirect,
    handleRecoverRagOnly,
    handleStopStream: () => stream.stop?.(),
    handleRetry,
    handleFeedback,
    handleClearChat,
    feedbackSubmitted,
    contextUsage,
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    showOptimisticSuggestion,
  };
}

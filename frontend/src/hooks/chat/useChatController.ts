import { AIMessage, HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { useStream } from "@langchain/react";
import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useChatBodyParams, type FlowMode } from "@/hooks/useChatBodyParams";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { getClientApiBase, toApiUrl } from "@/lib/api-base";
import { getMessageContent } from "@/lib/chat/messages";
import type { McpProgressEvent, McpToolInvocation } from "@/lib/types/chat";

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
  trace_id?: string;
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
  mcp_progress_events?: McpProgressEvent[];
  error?: string;
};

type MessageLike = {
  id?: string;
  role?: string;
  content?: string;
  parts?: { type?: string; text?: string; data?: unknown }[];
};
type PendingUserMessage = MessageLike & {
  submittedMessageCount: number;
};
type ChatStatus = "submitted" | "streaming" | "ready" | "error";
type SendOverrides = {
  mode?: FlowMode;
};
type SdkToolProgress = {
  toolCallId?: string;
  name?: string;
  state?: string;
  input?: unknown;
  data?: unknown;
  result?: unknown;
  error?: unknown;
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

type ChatStreamDebugEvent =
  | "submit"
  | "stop"
  | "error"
  | "status"
  | "stream.messages"
  | "stream.values"
  | "stream.toolProgress"
  | "visible.messages";

const CHAT_STREAM_DEBUG_FLAG = "rag_agent_debug_stream";
const CHAT_STREAM_DEBUG_BREAK_FLAG = "rag_agent_debug_stream_break";
const EMPTY_TOOL_PROGRESS: SdkToolProgress[] = [];

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

function isSameContextUsage(a: ContextUsage | null, b: ContextUsage): boolean {
  return (
    a?.tokens === b.tokens &&
    a.max === b.max &&
    a.percent === b.percent &&
    a.model_id === b.model_id
  );
}

function resolveLanggraphApiUrl(): string {
  const base = getClientApiBase();
  if (!base && typeof window !== "undefined") {
    return `${window.location.origin}/api/langgraph`;
  }
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
      Array.isArray(raw.mcp_progress_events) ||
      typeof raw.trace_id === "string" ||
      typeof raw.standalone_question === "string" ||
      typeof raw.error === "string" ||
      raw.mcp_used === true;
    if (!hasKnownReferenceField) continue;
    return {
      trace_id: typeof raw.trace_id === "string" ? raw.trace_id : undefined,
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
      mcp_progress_events: Array.isArray(raw.mcp_progress_events)
        ? (raw.mcp_progress_events as McpProgressEvent[])
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
      Array.isArray(raw.mcp_progress_events) ||
      typeof raw.trace_id === "string" ||
      typeof raw.standalone_question === "string" ||
      typeof raw.error === "string" ||
      raw.mcp_used === true;
    if (!hasKnownReferenceField) continue;
    return {
      trace_id: typeof raw.trace_id === "string" ? raw.trace_id : undefined,
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
      mcp_progress_events: Array.isArray(raw.mcp_progress_events)
        ? (raw.mcp_progress_events as McpProgressEvent[])
        : undefined,
      error: typeof raw.error === "string" ? raw.error : undefined,
    };
  }
  return null;
}

function toRoleFromRawValueMessage(rawMessage: Record<string, unknown>): MessageLike["role"] {
  const role = String(rawMessage.role ?? "").trim().toLowerCase();
  const type = String(rawMessage.type ?? "").trim().toLowerCase();
  if (role === "user" || role === "human" || type === "human") return "user";
  if (role === "assistant" || role === "ai" || type === "ai") return "assistant";
  if (role === "system" || type === "system") return "system";
  return undefined;
}

function toMessageFromRawValue(rawMessage: unknown, index: number): MessageLike | null {
  if (!rawMessage || typeof rawMessage !== "object") return null;
  const data = rawMessage as Record<string, unknown>;
  const role = toRoleFromRawValueMessage(data);
  if (!role) return null;

  const text = readText(data.content);
  const refData = toReferencesFromRawMessage(data);
  const parts: { type?: string; text?: string; data?: unknown }[] = [];
  if (text) parts.push({ type: "text", text });
  if (refData) parts.push({ type: "data-references", data: refData });

  return {
    id: typeof data.id === "string" ? data.id : `value-message-${index}`,
    role,
    content: text,
    parts,
  };
}

function hasVisibleUserMessageWithText(messages: MessageLike[], text: string): boolean {
  const normalizedText = text.trim();
  if (!normalizedText) return false;
  return messages.some(
    (message) =>
      message.role === "user" && getMessageContent(message).trim() === normalizedText,
  );
}

function traceIdFromMessage(message: MessageLike): string | undefined {
  const refPart = message.parts?.find((part) => part.type === "data-references");
  const data = refPart?.data;
  if (!data || typeof data !== "object") return undefined;
  const traceId = (data as ReferencePayload).trace_id;
  return typeof traceId === "string" && traceId.trim() ? traceId : undefined;
}

function referencePayloadFromMessage(message: MessageLike): ReferencePayload | null {
  const refPart = message.parts?.find((part) => part.type === "data-references");
  const data = refPart?.data;
  return data && typeof data === "object" ? (data as ReferencePayload) : null;
}

function stringifyToolPayload(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function toolProgressToMcpEvents(toolProgress: SdkToolProgress[]): McpProgressEvent[] {
  return toolProgress
    .map((tool): McpProgressEvent | null => {
      const toolName = typeof tool.name === "string" ? tool.name.trim() : "";
      if (!toolName) return null;
      const toolRunId =
        typeof tool.toolCallId === "string" && tool.toolCallId.trim()
          ? tool.toolCallId
          : undefined;
      const base = {
        tool_name: toolName,
        tool_run_id: toolRunId,
        args: tool.input,
      };
      if (tool.state === "completed") {
        return {
          ...base,
          phase: "end",
          result: stringifyToolPayload(tool.result),
        };
      }
      if (tool.state === "error") {
        return {
          ...base,
          phase: "error",
          error: stringifyToolPayload(tool.error),
        };
      }
      return {
        ...base,
        phase: "start",
        result: tool.state === "running" ? stringifyToolPayload(tool.data) : null,
      };
    })
    .filter((event): event is McpProgressEvent => event != null);
}

function withLiveToolProgress(
  messages: MessageLike[],
  progressEvents: McpProgressEvent[],
): MessageLike[] {
  if (progressEvents.length === 0) return messages;
  const toolsUsed = [...new Set(progressEvents.map((event) => event.tool_name))];
  const assistantIndex = [...messages]
    .map((message, index) => ({ message, index }))
    .reverse()
    .find(({ message }) => message.role === "assistant")?.index;
  const progressReferences = {
    citations: [],
    reranker_docs: [],
    mcp_used: true,
    mcp_tools_used: toolsUsed,
    mcp_progress_events: progressEvents,
  } satisfies ReferencePayload;

  if (assistantIndex == null) {
    const latestEvent = progressEvents.at(-1);
    return [
      ...messages,
      {
        id: `live-tool-progress-${latestEvent?.tool_run_id ?? latestEvent?.tool_name ?? "tool"}`,
        role: "assistant",
        content: "",
        parts: [{ type: "data-references", data: progressReferences }],
      },
    ];
  }

  const target = messages[assistantIndex];
  const currentReferences = referencePayloadFromMessage(target);
  const mergedReferences: ReferencePayload = {
    trace_id: currentReferences?.trace_id,
    standalone_question: currentReferences?.standalone_question,
    citations: currentReferences?.citations ?? [],
    reranker_docs: currentReferences?.reranker_docs ?? [],
    context_usage: currentReferences?.context_usage,
    mcp_used: currentReferences?.mcp_used ?? true,
    mcp_tools_used:
      currentReferences?.mcp_tools_used && currentReferences.mcp_tools_used.length > 0
        ? currentReferences.mcp_tools_used
        : toolsUsed,
    mcp_tool_invocations: currentReferences?.mcp_tool_invocations,
    mcp_progress_events: progressEvents,
    error: currentReferences?.error,
  };
  const next = [...messages];
  const nextParts = (target.parts ?? []).filter((part) => part.type !== "data-references");
  next[assistantIndex] = {
    ...target,
    parts: [...nextParts, { type: "data-references", data: mergedReferences }],
  };
  return next;
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

function hasRawUserMessageWithText(messages: BaseMessageWithKwargs[], text: string): boolean {
  const normalizedText = text.trim();
  if (!normalizedText) return false;
  return messages.some(
    (message) =>
      toRole(message) === "user" &&
      readText(message.content).trim() === normalizedText,
  );
}

function createPendingUserMessage(
  text: string,
  submittedMessageCount: number,
): PendingUserMessage {
  return {
    id: `pending-user-${Date.now()}`,
    role: "user",
    content: text,
    parts: [{ type: "text", text }],
    submittedMessageCount,
  };
}

function getDebugParam(name: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return new URLSearchParams(window.location.search).get(name);
  } catch {
    return null;
  }
}

function getDebugStorageValue(name: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(name);
  } catch {
    return null;
  }
}

function isTruthyDebugValue(value: string | null | undefined): boolean {
  if (!value) return false;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function isChatStreamDebugEnabled(): boolean {
  return (
    process.env.NEXT_PUBLIC_CHAT_STREAM_DEBUG === "true" ||
    isTruthyDebugValue(getDebugParam("debugStream")) ||
    isTruthyDebugValue(getDebugStorageValue(CHAT_STREAM_DEBUG_FLAG))
  );
}

function shouldBreakForChatStreamEvent(event: ChatStreamDebugEvent): boolean {
  const configured =
    getDebugParam("debugStreamBreak") ??
    getDebugStorageValue(CHAT_STREAM_DEBUG_BREAK_FLAG);
  if (!configured) return false;
  const normalized = configured.trim().toLowerCase();
  if (["1", "true", "yes", "on", "*", "all"].includes(normalized)) {
    return true;
  }
  return normalized
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .includes(event.toLowerCase());
}

function summarizeReferencePayload(refs: ReferencePayload | null): Record<string, unknown> | null {
  if (!refs) return null;
  return {
    trace_id: refs.trace_id,
    mcp_used: refs.mcp_used,
    mcp_tools_used: refs.mcp_tools_used,
    mcp_tool_invocations: refs.mcp_tool_invocations?.map((tool) => tool.tool_name),
    mcp_progress_events: refs.mcp_progress_events?.map((event) => ({
      phase: event.phase,
      tool_name: event.tool_name,
      tool_run_id: event.tool_run_id,
    })),
    citations: refs.citations?.length ?? 0,
    reranker_docs: refs.reranker_docs?.length ?? 0,
    error: refs.error,
  };
}

function summarizeBaseMessage(message: BaseMessageWithKwargs, index: number): Record<string, unknown> {
  const refs = toReferences(message);
  return {
    index,
    id: typeof message.id === "string" ? message.id : undefined,
    role: toRole(message),
    text: readText(message.content).slice(0, 240),
    refs: summarizeReferencePayload(refs),
  };
}

function summarizeVisibleMessage(message: MessageLike, index: number): Record<string, unknown> {
  const refPart = message.parts?.find((part) => part.type === "data-references");
  const refs =
    refPart?.data && typeof refPart.data === "object"
      ? (refPart.data as ReferencePayload)
      : null;
  return {
    index,
    id: message.id,
    role: message.role,
    text: getMessageContent(message).slice(0, 240),
    refs: summarizeReferencePayload(refs),
  };
}

function summarizeStreamValues(values: unknown): Record<string, unknown> {
  if (!values || typeof values !== "object") {
    return { type: typeof values };
  }
  const data = values as Record<string, unknown>;
  const rawMessages = Array.isArray(data.messages) ? data.messages : [];
  return {
    keys: Object.keys(data),
    message_count: rawMessages.length,
    messages: rawMessages.map((message, index) => {
      if (!message || typeof message !== "object") {
        return { index, type: typeof message };
      }
      const raw = message as Record<string, unknown>;
      return {
        index,
        id: raw.id,
        role: raw.role,
        type: raw.type,
        text: readText(raw.content).slice(0, 240),
        refs: summarizeReferencePayload(toReferencesFromRawMessage(raw)),
      };
    }),
  };
}

function summarizeToolProgress(toolProgress: SdkToolProgress[]): Record<string, unknown>[] {
  return toolProgress.map((tool, index) => ({
    index,
    toolCallId: tool.toolCallId,
    name: tool.name,
    state: tool.state,
    input: tool.input,
    data: tool.data,
    result: tool.result,
    error: tool.error,
  }));
}

function debugChatStream(
  event: ChatStreamDebugEvent,
  payload: Record<string, unknown>,
): void {
  if (!isChatStreamDebugEnabled()) return;
  const time = new Date().toISOString();
  console.groupCollapsed(`[chat-stream] ${event} ${time}`);
  console.log(payload);
  console.groupEnd();
  if (shouldBreakForChatStreamEvent(event)) {
    debugger;
  }
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
  const [, setFeedbackSubmitted] = useState(false);
  const [feedbackSubmittedMessageIndexes, setFeedbackSubmittedMessageIndexes] = useState<
    Set<number>
  >(() => new Set());
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [pendingUserMessage, setPendingUserMessage] = useState<PendingUserMessage | null>(null);
  const lastErrorToastKeyRef = useRef<string | null>(null);

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
      debugChatStream("stop", { threadId });
      toast.success("Generation stopped");
    },
    onError: (error) => {
      debugChatStream("error", { threadId, error });
      console.error("Chat error:", error);
      const message = error instanceof Error ? error.message : String(error);
      toast.error(message);
    },
  });

  const streamMessages = stream.messages;
  const streamMessageCount = streamMessages?.length ?? 0;
  const streamValues = (stream as { values?: unknown }).values;
  const streamToolProgress =
    (stream as { toolProgress?: SdkToolProgress[] }).toolProgress ?? EMPTY_TOOL_PROGRESS;
  const liveToolProgressEvents = useMemo(
    () => toolProgressToMcpEvents(streamToolProgress),
    [streamToolProgress],
  );

  const sendUserMessage = useCallback(
    (text: string, overrides?: SendOverrides) => {
      const trimmedText = text.trim();
      if (!trimmedText) return;

      const effectiveMode = overrides?.mode ?? bodyParams.mode;
      setPendingUserMessage(createPendingUserMessage(trimmedText, streamMessageCount));
      debugChatStream("submit", {
        threadId,
        text: trimmedText,
        streamMessageCount,
        bodyParams,
        effectiveMode,
      });
      void Promise.resolve(
        stream.submit({
          messages: [{ type: "human", content: trimmedText }],
          model: bodyParams.model,
          session_id: bodyParams.session_id,
          collection_name: bodyParams.collection_name,
          enable_reranker: bodyParams.enable_reranker,
          enable_tracing: bodyParams.enable_tracing,
          mode: effectiveMode,
          context: { ...bodyParams, mode: effectiveMode },
          metadata: { ...bodyParams, mode: effectiveMode },
          configurable: { ...bodyParams, mode: effectiveMode },
        }, { streamMode: ["values", "tools"] }),
      ).catch(() => undefined);
    },
    [bodyParams, stream, streamMessageCount, threadId],
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

    const rawValues = streamValues;
    const valueMessages =
      rawValues && typeof rawValues === "object" && Array.isArray((rawValues as { messages?: unknown }).messages)
        ? ((rawValues as { messages: unknown[] }).messages as unknown[])
        : [];
    const valueMapped = valueMessages
      .map((message, index) => toMessageFromRawValue(message, index))
      .filter((message): message is MessageLike => message != null);

    const hasValueProgress = valueMapped.some((message) => {
      const refPart = message.parts?.find((part) => part.type === "data-references");
      const data = refPart?.data;
      return (
        data != null &&
        typeof data === "object" &&
        Array.isArray((data as ReferencePayload).mcp_progress_events) &&
        ((data as ReferencePayload).mcp_progress_events?.length ?? 0) > 0
      );
    });
    const baseMapped =
      valueMapped.length > mapped.length || hasValueProgress ? valueMapped : mapped;

    // Fallback for cases where class-message metadata drops reference payloads.
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
    const pendingText = getMessageContent(pendingUserMessage);
    const hasStreamedPendingUser =
      pendingUserMessage != null &&
      (hasVisibleUserMessageWithText(baseMapped, pendingText) ||
        hasRawUserMessageWithText(
          raw.slice(pendingUserMessage.submittedMessageCount),
          pendingText,
        ));
    const visibleMessages =
      pendingUserMessage && !hasStreamedPendingUser
        ? [...baseMapped, pendingUserMessage]
        : baseMapped;

    if (!fallbackRefs) {
      return withLiveToolProgress(visibleMessages, liveToolProgressEvents);
    }

    const lastAssistantIdx = [...visibleMessages]
      .map((msg, idx) => ({ msg, idx }))
      .reverse()
      .find(({ msg }) => msg.role === "assistant")?.idx;
    if (lastAssistantIdx == null) {
      return withLiveToolProgress(visibleMessages, liveToolProgressEvents);
    }
    const target = visibleMessages[lastAssistantIdx];
    const hasRefsAlready =
      Array.isArray(target.parts) &&
      target.parts.some((part) => part?.type === "data-references" && part.data != null);
    if (hasRefsAlready) {
      return withLiveToolProgress(visibleMessages, liveToolProgressEvents);
    }

    const next = [...visibleMessages];
    next[lastAssistantIdx] = {
      ...target,
      parts: [...(target.parts ?? []), { type: "data-references", data: fallbackRefs }],
    };
    return withLiveToolProgress(next, liveToolProgressEvents);
  }, [liveToolProgressEvents, pendingUserMessage, streamMessages, streamValues]);

  const rawStreamStatus = (stream as { status?: unknown }).status;
  const status = normalizeStatus(
    rawStreamStatus,
    stream.isLoading,
    stream.error != null,
  );

  useEffect(() => {
    debugChatStream("status", {
      threadId,
      status,
      rawStatus: rawStreamStatus,
      isLoading: stream.isLoading,
      hasError: stream.error != null,
      error: stream.error,
    });
  }, [rawStreamStatus, status, stream.error, stream.isLoading, threadId]);

  useEffect(() => {
    const raw = (streamMessages ?? []) as BaseMessageWithKwargs[];
    debugChatStream("stream.messages", {
      threadId,
      count: raw.length,
      messages: raw.map(summarizeBaseMessage),
      raw,
    });
  }, [streamMessages, threadId]);

  useEffect(() => {
    debugChatStream("stream.values", {
      threadId,
      summary: summarizeStreamValues(streamValues),
      raw: streamValues,
    });
  }, [streamValues, threadId]);

  useEffect(() => {
    debugChatStream("stream.toolProgress", {
      threadId,
      count: streamToolProgress.length,
      progress: summarizeToolProgress(streamToolProgress),
      mcpProgressEvents: liveToolProgressEvents,
    });
  }, [liveToolProgressEvents, streamToolProgress, threadId]);

  useEffect(() => {
    debugChatStream("visible.messages", {
      threadId,
      count: messages.length,
      messages: messages.map(summarizeVisibleMessage),
      raw: messages,
    });
  }, [messages, threadId]);

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
        setContextUsage((previous) =>
          isSameContextUsage(previous, contextUsagePayload) ? previous : contextUsagePayload,
        );
      });
    }
    if (typeof refs.error === "string" && refs.error.length > 0) {
      const errorToastKey = `${lastAssistant.id ?? "assistant"}:${refs.error}`;
      if (lastErrorToastKeyRef.current === errorToastKey) return;
      lastErrorToastKeyRef.current = errorToastKey;
      toast.error(refs.error, "Search unavailable");
    }
  }, [messages, toast]);

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
    async (stars: number, messageIndex: number) => {
      const lastAssistant = messages[messageIndex] as MessageLike | undefined;
      if (lastAssistant?.role !== "assistant") return;
      const lastUser = [...messages.slice(0, messageIndex)]
        .reverse()
        .find((message) => message.role === "user");
      if (!lastUser) return;

      const question = getMessageContent(lastUser);
      const answer = getMessageContent(lastAssistant);
      const traceId = traceIdFromMessage(lastAssistant);

      try {
        const res = await fetch(toApiUrl("/api/feedback"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, answer, feedback: stars, trace_id: traceId }),
        });
        if (res.ok) {
          setFeedbackSubmitted(true);
          setFeedbackSubmittedMessageIndexes((previous) => {
            const next = new Set(previous);
            next.add(messageIndex);
            return next;
          });
        }
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
    setPendingUserMessage(null);
    setFeedbackSubmittedMessageIndexes(new Set());
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
    feedbackSubmittedMessageIndexes,
    contextUsage,
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    showOptimisticSuggestion,
  };
}

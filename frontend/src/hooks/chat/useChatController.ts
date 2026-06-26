import { useStream, type AssembledToolCall } from "@langchain/react";
import {
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type MessageLike,
  type UseChatControllerArgs,
} from "@/hooks/chat/controller-types";
import {
  getLastUserMessageText,
  normalizeStatus,
  projectStreamMessages,
} from "@/hooks/chat/message-projection";
import {
  type BaseMessageWithKwargs,
  isSameContextUsage,
} from "@/hooks/chat/references";
import {
  debugChatStream,
  summarizeBaseMessage,
  summarizeToolProgress,
  summarizeVisibleMessage,
} from "@/hooks/chat/stream-debug";
import { resolveLanggraphApiUrl } from "@/hooks/chat/stream-config";
import { toolCallsToMcpEvents, type SdkToolProgress } from "@/hooks/chat/tool-progress";
import { useChatActions } from "@/hooks/chat/useChatActions";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useChatBodyParams } from "@/hooks/useChatBodyParams";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import type { ContextUsage } from "@/lib/types/chat";

const EMPTY_TOOL_CALLS: AssembledToolCall[] = [];

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
  });

  const streamMessages = stream.messages;
  const streamToolCalls = stream.toolCalls ?? EMPTY_TOOL_CALLS;
  const liveToolProgressEvents = useMemo(
    () => toolCallsToMcpEvents(streamToolCalls),
    [streamToolCalls],
  );

  const messages = useMemo(
    () =>
      projectStreamMessages({
        streamMessages: streamMessages as BaseMessageWithKwargs[] | undefined,
        liveToolProgressEvents,
      }),
    [liveToolProgressEvents, streamMessages],
  );

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
    debugChatStream("stream.toolProgress", {
      threadId,
      count: streamToolCalls.length,
      progress: summarizeToolProgress(
        streamToolCalls.map((tool) => ({
          toolCallId: tool.callId,
          name: tool.name,
          state: tool.status,
          input: tool.input,
          result: tool.output,
          error: tool.error,
        })),
      ),
      mcpProgressEvents: liveToolProgressEvents,
    });
  }, [liveToolProgressEvents, streamToolCalls, threadId]);

  useEffect(() => {
    if (stream.error == null) return;
    debugChatStream("error", { threadId, error: stream.error });
    console.error("Chat error:", stream.error);
    const message =
      stream.error instanceof Error ? stream.error.message : String(stream.error);
    const errorToastKey = `stream:${threadId}:${message}`;
    if (lastErrorToastKeyRef.current === errorToastKey) return;
    lastErrorToastKeyRef.current = errorToastKey;
    toast.error(message);
  }, [stream.error, threadId, toast]);

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
    const refs = lastAssistant?.references;
    if (!refs) return;
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
    handleClearChat,
    handleFeedback,
    handleRecoverDirect,
    handleRecoverRagOnly,
    handleResumeTurn,
    handleRetry,
    handleStopStream,
    handleSubmit,
    sendUserMessage,
  } = useChatActions({
    bodyParams,
    clearSessionChat,
    input,
    messages,
    setContextUsage,
    setFeedbackSubmitted,
    setFeedbackSubmittedMessageIndexes,
    setInput,
    setMaxCitationsToShow,
    stream,
    threadId,
    toast,
  });
  const {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
  } = useSuggestions({
    messages,
    status,
    sendMessage: (text) => sendUserMessage(text),
    selectedModel,
    setFeedbackSubmitted,
  });

  const chatContainerRef = useScrollToBottom(status, messages);

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
    handleStopStream,
    handleRetry,
    handleFeedback,
    handleClearChat,
    feedbackSubmittedMessageIndexes,
    contextUsage,
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
  };
}

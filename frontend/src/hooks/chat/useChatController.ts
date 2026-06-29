import type { AssembledToolCall } from "@langchain/react";
import {
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type UseChatControllerArgs,
} from "@/hooks/chat/controller-types";
import {
  getLastUserMessageText,
  normalizeStatus,
  projectStreamMessages,
} from "@/hooks/chat/message-projection";
import { isMissingThreadError } from "@/hooks/chat/thread-errors";
import {
  type BaseMessageWithKwargs,
  isSameContextUsage,
} from "@/hooks/chat/references";
import { useChatActions } from "@/hooks/chat/useChatActions";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useChatBodyParams } from "@/hooks/useChatBodyParams";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { useLangGraphStream } from "@/providers/langgraph-stream-provider";
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
  const lastRecoveredMissingThreadKeyRef = useRef<string | null>(null);

  const bodyParams = useChatBodyParams({
    selectedModel,
    threadId,
    sessionId,
    collectionName,
    enableReranker,
    enableTracing,
    flowMode,
  });

  const { stream } = useLangGraphStream();

  const streamMessages = stream.messages;
  const streamToolCalls = stream.toolCalls ?? EMPTY_TOOL_CALLS;

  const messages = useMemo(
    () =>
      projectStreamMessages({
        streamMessages: streamMessages as BaseMessageWithKwargs[] | undefined,
      }),
    [streamMessages],
  );

  const rawStreamStatus = (stream as { status?: unknown }).status;
  const status = normalizeStatus(
    rawStreamStatus,
    stream.isLoading,
    stream.error != null,
  );

  useEffect(() => {
    if (stream.error == null) return;
    console.error("Chat error:", stream.error);
    const message =
      stream.error instanceof Error ? stream.error.message : String(stream.error);
    if (isMissingThreadError(stream.error, threadId)) {
      const recoveryKey = `missing-thread:${threadId}:${message}`;
      if (lastRecoveredMissingThreadKeyRef.current === recoveryKey) return;
      lastRecoveredMissingThreadKeyRef.current = recoveryKey;
      clearSessionChat({
        setFeedbackSubmitted,
        setContextUsage,
      });
      return;
    }
    const errorToastKey = `stream:${threadId}:${message}`;
    if (lastErrorToastKeyRef.current === errorToastKey) return;
    lastErrorToastKeyRef.current = errorToastKey;
    toast.error(message);
  }, [clearSessionChat, setContextUsage, setFeedbackSubmitted, stream.error, threadId, toast]);

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
    toolCalls: streamToolCalls,
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

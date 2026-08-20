import { useMessageMetadata } from "@langchain/react";
import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import type { UseChatControllerArgs } from "@/hooks/chat/controller-types";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";
import {
  getLastUserMessageText,
  normalizeStatus,
  projectStreamMessages,
  selectMessagesForStatus,
} from "@/hooks/chat/message-projection";
import {
  type BaseMessageWithKwargs,
  isSameContextUsage,
} from "@/hooks/chat/references";
import { isMissingThreadError } from "@/hooks/chat/thread-errors";
import {
  deriveToolCallsFromMessages,
  filterToolCallsForChatStatus,
  type NativeToolCall,
} from "@/hooks/chat/tool-call-mapping";
import { useChatActions } from "@/hooks/chat/useChatActions";
import { useChatBodyParams } from "@/hooks/useChatBodyParams";
import { useSuggestions } from "@/hooks/useSuggestions";
import type { ContextUsage } from "@/lib/types/chat";
import { useLangGraphStream } from "@/providers/langgraph-stream-provider";

const EMPTY_TOOL_CALLS: NativeToolCall[] = [];

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
  removeThreadHistoryEntry,
}: UseChatControllerArgs) {
  const [input, setInput] = useState("");
  const [maxCitationsToShow, setMaxCitationsToShow] = useState(10);
  const [, setFeedbackSubmitted] = useState(false);
  const [feedbackSubmittedMessageIndexes, setFeedbackSubmittedMessageIndexes] =
    useState<Set<number>>(() => new Set());
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [submitError, setSubmitError] = useState<Error | null>(null);
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

  const { stream, transportError } = useLangGraphStream();
  const effectiveThreadId = threadId ?? stream.threadId ?? null;

  const streamMessages = stream.messages;
  const stateMessages = (stream.values as { messages?: unknown } | undefined)
    ?.messages;
  const streamToolCalls = stream.toolCalls ?? EMPTY_TOOL_CALLS;
  const toolCallsFromMessages = useMemo(
    () => deriveToolCallsFromMessages(streamMessages as BaseMessageWithKwargs[]),
    [streamMessages]
  );
  const resolvedToolCalls = useMemo((): NativeToolCall[] => {
    if (streamToolCalls.length > 0) {
      debugChatStage("resolvedToolCalls", {
        source: "streamToolCalls",
        count: streamToolCalls.length,
        names: streamToolCalls.map((tc) => tc.name),
      });
      return streamToolCalls;
    }
    if (toolCallsFromMessages.length > 0) {
      debugChatStage("resolvedToolCalls", {
        source: "toolCallsFromMessages",
        count: toolCallsFromMessages.length,
        names: toolCallsFromMessages.map((tc) => tc.name),
        callIds: toolCallsFromMessages.map((tc) => tc.callId ?? (tc as Record<string, unknown>).id),
      });
      return toolCallsFromMessages;
    }
    debugChatStage("resolvedToolCalls", { source: "empty" });
    return EMPTY_TOOL_CALLS;
  }, [streamToolCalls, toolCallsFromMessages]);
  const progress =
    typeof (stream.values as { progress?: unknown } | undefined)?.progress ===
    "string"
      ? (stream.values as { progress: string }).progress
      : undefined;

  const rawStreamStatus = (stream as { status?: unknown }).status;
  const status = normalizeStatus(
    rawStreamStatus,
    stream.isLoading,
    stream.error != null || submitError != null || transportError != null
  );
  const visibleToolCalls = useMemo(
    () => filterToolCallsForChatStatus(resolvedToolCalls, status),
    [resolvedToolCalls, status]
  );

  const messages = useMemo(() => {
    const liveMessages = projectStreamMessages({
      streamMessages: streamMessages as BaseMessageWithKwargs[] | undefined,
    });
    const finalizedMessages = Array.isArray(stateMessages)
      ? projectStreamMessages({
          streamMessages: stateMessages as BaseMessageWithKwargs[],
        })
      : undefined;
    const selectedMessages = selectMessagesForStatus(
      liveMessages,
      finalizedMessages,
      status
    );
    debugChatStage("selectMessagesForStatus", {
      status,
      live: summarizeMessages(liveMessages),
      finalized: finalizedMessages
        ? summarizeMessages(finalizedMessages)
        : undefined,
      selected: summarizeMessages(selectedMessages),
    });
    return selectedMessages;
  }, [stateMessages, status, streamMessages]);
  const lastUserMessageId = useMemo(
    () =>
      [...messages].reverse().find((message) => message.role === "user")?.id,
    [messages]
  );
  const retryCheckpointId = useMessageMetadata(
    stream,
    lastUserMessageId
  )?.parentCheckpointId;

  useEffect(() => {
    if (stream.error == null) {
      return;
    }
    console.error("Chat error:", stream.error);
    const message =
      stream.error instanceof Error
        ? stream.error.message
        : String(stream.error);
    if (isMissingThreadError(stream.error, threadId)) {
      const recoveryKey = `missing-thread:${threadId}:${message}`;
      if (lastRecoveredMissingThreadKeyRef.current === recoveryKey) {
        return;
      }
      lastRecoveredMissingThreadKeyRef.current = recoveryKey;
      clearSessionChat({
        setFeedbackSubmitted,
        setContextUsage,
      });
      return;
    }
    const errorToastKey = `stream:${threadId}:${message}`;
    if (lastErrorToastKeyRef.current === errorToastKey) {
      return;
    }
    lastErrorToastKeyRef.current = errorToastKey;
    toast.error(message);
  }, [
    clearSessionChat,
    setContextUsage,
    setFeedbackSubmitted,
    stream.error,
    threadId,
    toast,
  ]);

  useEffect(() => {
    if (submitError == null) {
      return;
    }
    const message = submitError.message || "Chat request failed";
    const errorToastKey = `submit:${threadId}:${message}`;
    if (lastErrorToastKeyRef.current === errorToastKey) {
      return;
    }
    lastErrorToastKeyRef.current = errorToastKey;
    toast.error(message);
  }, [submitError, threadId, toast]);

  useEffect(() => {
    if (transportError == null) {
      return;
    }
    const message = transportError.message || "Chat request failed";
    if (isMissingThreadError(transportError, threadId)) {
      const recoveryKey = `missing-thread-transport:${threadId}:${message}`;
      if (lastRecoveredMissingThreadKeyRef.current === recoveryKey) {
        return;
      }
      lastRecoveredMissingThreadKeyRef.current = recoveryKey;
      clearSessionChat({
        setFeedbackSubmitted,
        setContextUsage,
      });
      return;
    }
    const errorToastKey = `transport:${threadId}:${message}`;
    if (lastErrorToastKeyRef.current === errorToastKey) {
      return;
    }
    lastErrorToastKeyRef.current = errorToastKey;
    toast.error(message);
  }, [
    clearSessionChat,
    setContextUsage,
    setFeedbackSubmitted,
    threadId,
    toast,
    transportError,
  ]);

  useEffect(() => {
    const lastAssistant = [...messages]
      .reverse()
      .find((msg) => msg.role === "assistant");
    const refs = lastAssistant?.references;
    if (!refs) {
      return;
    }
    const contextUsagePayload = refs.context_usage;
    if (contextUsagePayload) {
      startTransition(() => {
        setContextUsage((previous) =>
          isSameContextUsage(previous, contextUsagePayload)
            ? previous
            : contextUsagePayload
        );
      });
    }
    if (typeof refs.error === "string" && refs.error.length > 0) {
      const errorToastKey = `${lastAssistant.id ?? "assistant"}:${refs.error}`;
      if (lastErrorToastKeyRef.current === errorToastKey) {
        return;
      }
      lastErrorToastKeyRef.current = errorToastKey;
      toast.error(refs.error, "Search unavailable");
    }
  }, [messages, toast]);

  const {
    handleClearChat,
    handleDeleteThread,
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
    removeThreadHistoryEntry,
    setContextUsage,
    setFeedbackSubmitted,
    setFeedbackSubmittedMessageIndexes,
    setInput,
    setMaxCitationsToShow,
    setSubmitError,
    stream,
    threadId,
    retryCheckpointId,
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
    threadId: effectiveThreadId,
    setFeedbackSubmitted,
  });

  const canStopStream = status === "submitted" || status === "streaming";
  const canResumeTurn =
    status === "error" && getLastUserMessageText(messages).length > 0;

  return {
    input,
    setInput,
    messages,
    progress,
    toolCalls: visibleToolCalls,
    status,
    maxCitationsToShow,
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
    handleDeleteThread,
    feedbackSubmittedMessageIndexes,
    contextUsage,
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
  };
}

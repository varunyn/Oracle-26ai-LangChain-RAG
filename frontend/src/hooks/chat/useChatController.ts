import { useMessageMetadata } from "@langchain/react";
import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import type {
  ChatStatus,
  UseChatControllerArgs,
} from "@/hooks/chat/controller-types";
import {
  getLastUserMessageText,
  projectStreamMessages,
} from "@/hooks/chat/message-projection";
import {
  type BaseMessageWithKwargs,
  isSameContextUsage,
} from "@/hooks/chat/references";
import { isMissingThreadError } from "@/hooks/chat/thread-errors";
import {
  filterToolCallsForChatStatus,
  hydrateToolCallsFromMessages,
  mergeHydratedAndLiveToolCalls,
  type NativeToolCall,
} from "@/hooks/chat/tool-call-mapping";
import { useChatActions } from "@/hooks/chat/useChatActions";
import { useChatBodyParams } from "@/hooks/useChatBodyParams";
import { useSuggestions } from "@/hooks/useSuggestions";
import type { ContextUsage } from "@/lib/types/chat";
import { useLangGraphStream } from "@/providers/langgraph-stream-provider";

const EMPTY_TOOL_CALLS: NativeToolCall[] = [];

export function resolveChatStatus(args: {
  hasError: boolean;
  isThreadLoading: boolean;
  isLoading: boolean;
  reconnecting: boolean;
}): ChatStatus {
  if (args.hasError) {
    return "failed";
  }
  if (args.isThreadLoading) {
    return "hydrating";
  }
  if (args.reconnecting) {
    return "reconnecting";
  }
  return args.isLoading ? "running" : "idle";
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
  refreshThreadHistory,
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

  const { reconnect, stream } = useLangGraphStream();
  const effectiveThreadId = threadId ?? stream.threadId ?? null;

  const streamMessages = stream.messages;
  const streamToolCalls = stream.toolCalls ?? EMPTY_TOOL_CALLS;
  const progress = stream.values.progress;
  const status = resolveChatStatus({
    hasError: stream.error != null,
    isThreadLoading: stream.isThreadLoading,
    isLoading: stream.isLoading,
    reconnecting: reconnect != null,
  });
  const hydratedToolCalls = useMemo(
    () => hydrateToolCallsFromMessages(streamMessages),
    [streamMessages]
  );
  const resolvedToolCalls = useMemo(
    () =>
      mergeHydratedAndLiveToolCalls(
        hydratedToolCalls,
        streamToolCalls as NativeToolCall[]
      ),
    [hydratedToolCalls, streamToolCalls]
  );
  const visibleToolCalls = useMemo(
    () => filterToolCallsForChatStatus(resolvedToolCalls, status),
    [resolvedToolCalls, status]
  );

  const messages = useMemo(
    () => projectStreamMessages(streamMessages as BaseMessageWithKwargs[]),
    [streamMessages]
  );
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
  }, [clearSessionChat, stream.error, threadId, toast]);

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
    refreshThreadHistory,
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

  const canStopStream = status === "running" || status === "reconnecting";
  const canResumeTurn =
    status === "failed" && getLastUserMessageText(messages).length > 0;

  return {
    input,
    setInput,
    messages,
    progress,
    toolCalls: visibleToolCalls,
    status,
    reconnect,
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

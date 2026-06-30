import type { useStream } from "@langchain/react";
import type { Dispatch, SetStateAction, SyntheticEvent } from "react";
import { useCallback } from "react";
import type {
  ClearSessionChat,
  MessageLike,
  RemoveThreadHistoryEntry,
  SendOverrides,
  ToastApi,
} from "@/hooks/chat/controller-types";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";
import { getLastUserMessageText } from "@/hooks/chat/message-projection";
import { traceIdFromMessage } from "@/hooks/chat/references";
import {
  buildLangGraphSubmitPayload,
  type ChatBodyParams,
} from "@/hooks/chat/stream-config";
import { toApiUrl } from "@/lib/api-base";
import { getMessageContent } from "@/lib/chat/messages";
import type { ContextUsage } from "@/lib/types/chat";

type StreamType = ReturnType<typeof useStream>;
type SubmitOptions = Parameters<StreamType["submit"]>[1];

export function useChatActions(args: {
  bodyParams: ChatBodyParams;
  clearSessionChat: ClearSessionChat;
  input: string;
  messages: MessageLike[];
  removeThreadHistoryEntry: RemoveThreadHistoryEntry;
  setContextUsage: Dispatch<SetStateAction<ContextUsage | null>>;
  setFeedbackSubmitted: Dispatch<SetStateAction<boolean>>;
  setFeedbackSubmittedMessageIndexes: Dispatch<SetStateAction<Set<number>>>;
  setInput: Dispatch<SetStateAction<string>>;
  setMaxCitationsToShow: Dispatch<SetStateAction<number>>;
  setSubmitError: Dispatch<SetStateAction<Error | null>>;
  stream: StreamType;
  threadId: string | null;
  toast: ToastApi;
}): {
  sendUserMessage: (text: string, overrides?: SendOverrides) => void;
  handleSubmit: (e: SyntheticEvent<HTMLFormElement>) => void;
  handleRetry: () => void;
  handleRecoverDirect: () => void;
  handleRecoverRagOnly: () => void;
  handleResumeTurn: () => void;
  handleStopStream: () => void;
  handleFeedback: (stars: number, messageIndex: number) => Promise<void>;
  handleClearChat: () => Promise<void>;
  handleDeleteThread: (threadId: string) => Promise<void>;
} {
  const {
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
    toast,
  } = args;

  const sendUserMessage = useCallback(
    (text: string, overrides?: SendOverrides) => {
      const trimmedText = text.trim();
      if (!trimmedText) {
        return;
      }

      const effectiveMode = overrides?.mode ?? bodyParams.mode;
      const payload = buildLangGraphSubmitPayload(trimmedText, {
        ...bodyParams,
        mode: effectiveMode,
      });
      debugChatStage("sendUserMessage", {
        threadId,
        effectiveMode,
        inputPreview: trimmedText.slice(0, 120),
        visibleMessages: summarizeMessages(messages),
        payloadInputCount: Array.isArray(payload.input.messages)
          ? payload.input.messages.length
          : 0,
      });
      const submitOptions: SubmitOptions = {
        config: payload.config,
      };
      setSubmitError(null);
      void Promise.resolve(stream.submit(payload.input, submitOptions)).catch(
        (error: unknown) => {
          setSubmitError(
            error instanceof Error ? error : new Error(String(error))
          );
        }
      );
    },
    [bodyParams, messages, setSubmitError, stream, threadId]
  );

  const handleSubmit = useCallback(
    (e: SyntheticEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!input.trim()) {
        return;
      }
      setFeedbackSubmitted(false);
      sendUserMessage(input);
      setInput("");
      setMaxCitationsToShow(10);
    },
    [
      input,
      sendUserMessage,
      setFeedbackSubmitted,
      setInput,
      setMaxCitationsToShow,
    ]
  );

  const handleRetry = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) {
      return;
    }
    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleRecoverDirect = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) {
      return;
    }
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "direct" });
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleRecoverRagOnly = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) {
      return;
    }
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "rag" });
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleResumeTurn = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) {
      return;
    }
    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleStopStream = useCallback(() => {
    void stream.stop?.();
    toast.success("Generation stopped");
  }, [stream, toast]);

  const handleFeedback = useCallback(
    async (stars: number, messageIndex: number) => {
      const lastAssistant = messages[messageIndex] as MessageLike | undefined;
      if (lastAssistant?.role !== "assistant") {
        return;
      }
      const lastUser = [...messages.slice(0, messageIndex)]
        .reverse()
        .find((message) => message.role === "user");
      if (!lastUser) {
        return;
      }

      const question = getMessageContent(lastUser);
      const answer = getMessageContent(lastAssistant);
      const traceId = traceIdFromMessage(lastAssistant);

      try {
        const res = await fetch(toApiUrl("/api/feedback"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            answer,
            feedback: stars,
            trace_id: traceId,
          }),
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
    [messages, setFeedbackSubmitted, setFeedbackSubmittedMessageIndexes, toast]
  );

  const handleClearChat = useCallback(async () => {
    debugChatStage("clearChat.start", { threadId });
    try {
      if (typeof stream.stop === "function") {
        await stream.stop();
      }
      if (threadId) {
        await stream.client.threads.delete(threadId);
      }
    } catch (error) {
      console.error("Thread cleanup failed:", error);
      debugChatStage("clearChat.failed", { threadId, error: String(error) });
      toast.error("Could not clear chat history");
      return;
    }

    clearSessionChat({
      threadId,
      setFeedbackSubmitted,
      setContextUsage,
    });
    setFeedbackSubmittedMessageIndexes(new Set());
    debugChatStage("clearChat.succeeded", { threadId });
    toast.success("Chat cleared");
  }, [
    clearSessionChat,
    setContextUsage,
    setFeedbackSubmitted,
    setFeedbackSubmittedMessageIndexes,
    stream,
    threadId,
    toast,
  ]);

  const handleDeleteThread = useCallback(
    async (targetThreadId: string) => {
      const normalizedThreadId = targetThreadId.trim();
      if (!normalizedThreadId) {
        return;
      }
      debugChatStage("deleteThread.start", {
        activeThreadId: threadId,
        targetThreadId: normalizedThreadId,
      });
      if (normalizedThreadId === threadId) {
        await handleClearChat();
        return;
      }
      try {
        await stream.client.threads.delete(normalizedThreadId);
      } catch (error) {
        console.error("Thread delete failed:", error);
        debugChatStage("deleteThread.failed", {
          activeThreadId: threadId,
          targetThreadId: normalizedThreadId,
          error: String(error),
        });
        toast.error("Could not delete chat");
        return;
      }

      removeThreadHistoryEntry(normalizedThreadId);

      debugChatStage("deleteThread.succeeded", {
        activeThreadId: threadId,
        targetThreadId: normalizedThreadId,
      });
      toast.success("Chat deleted");
    },
    [
      clearSessionChat,
      removeThreadHistoryEntry,
      handleClearChat,
      stream,
      threadId,
      toast,
    ]
  );

  return {
    sendUserMessage,
    handleSubmit,
    handleRetry,
    handleRecoverDirect,
    handleRecoverRagOnly,
    handleResumeTurn,
    handleStopStream,
    handleFeedback,
    handleClearChat,
    handleDeleteThread,
  };
}

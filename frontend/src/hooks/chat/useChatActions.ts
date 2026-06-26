import type { Dispatch, SetStateAction, SyntheticEvent } from "react";
import { useCallback } from "react";
import { useStream } from "@langchain/react";
import type { ContextUsage } from "@/lib/types/chat";
import { toApiUrl } from "@/lib/api-base";
import { getMessageContent } from "@/lib/chat/messages";
import type {
  ClearSessionChat,
  MessageLike,
  SendOverrides,
  ToastApi,
} from "@/hooks/chat/controller-types";
import { buildSubmitPayload, type ChatBodyParams } from "@/hooks/chat/stream-config";
import { getLastUserMessageText } from "@/hooks/chat/message-projection";
import { traceIdFromMessage } from "@/hooks/chat/references";
import { debugChatStream } from "@/hooks/chat/stream-debug";

type StreamType = ReturnType<typeof useStream>;
type SubmitOptions = Parameters<StreamType["submit"]>[1] & {
  optimisticValues?: (previous: { messages?: unknown }) => { messages: unknown[] };
};

function createClientMessageId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function useChatActions(args: {
  bodyParams: ChatBodyParams;
  clearSessionChat: ClearSessionChat;
  input: string;
  messages: MessageLike[];
  setContextUsage: Dispatch<SetStateAction<ContextUsage | null>>;
  setFeedbackSubmitted: Dispatch<SetStateAction<boolean>>;
  setFeedbackSubmittedMessageIndexes: Dispatch<SetStateAction<Set<number>>>;
  setInput: Dispatch<SetStateAction<string>>;
  setMaxCitationsToShow: Dispatch<SetStateAction<number>>;
  stream: StreamType;
  threadId: string;
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
} {
  const {
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
  } = args;

  const sendUserMessage = useCallback(
    (text: string, overrides?: SendOverrides) => {
      const trimmedText = text.trim();
      if (!trimmedText) return;

      const effectiveMode = overrides?.mode ?? bodyParams.mode;
      debugChatStream("submit", {
        threadId,
        text: trimmedText,
        bodyParams,
        effectiveMode,
      });
      const userMessageId = createClientMessageId();
      const submitOptions: SubmitOptions = {
        optimisticValues: (previous) => ({
          messages: [
            ...(Array.isArray(previous.messages) ? previous.messages : []),
            { id: userMessageId, type: "human", content: trimmedText },
          ],
        }),
      };
      void Promise.resolve(
        stream.submit(
          buildSubmitPayload(trimmedText, bodyParams, effectiveMode, userMessageId),
          submitOptions,
        ),
      ).catch(() => undefined);
    },
    [bodyParams, stream, threadId],
  );

  const handleSubmit = useCallback(
    (e: SyntheticEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!input.trim()) return;
      setFeedbackSubmitted(false);
      sendUserMessage(input);
      setInput("");
      setMaxCitationsToShow(10);
    },
    [input, sendUserMessage, setFeedbackSubmitted, setInput, setMaxCitationsToShow],
  );

  const handleRetry = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleRecoverDirect = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "direct" });
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleRecoverRagOnly = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text, { mode: "rag" });
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleResumeTurn = useCallback(() => {
    const text = getLastUserMessageText(messages);
    if (!text) return;
    setFeedbackSubmitted(false);
    sendUserMessage(text);
  }, [messages, sendUserMessage, setFeedbackSubmitted]);

  const handleStopStream = useCallback(() => {
    debugChatStream("stop", { threadId });
    void stream.stop?.();
    toast.success("Generation stopped");
  }, [stream, threadId, toast]);

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
    [messages, setFeedbackSubmitted, setFeedbackSubmittedMessageIndexes, toast],
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
    setFeedbackSubmittedMessageIndexes(new Set());
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
  };
}

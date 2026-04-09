import { useChat } from "@ai-sdk/react";
import { useCallback, useState } from "react";
import { useSuggestions } from "@/hooks/useSuggestions";
import { useChatBodyParams, type FlowMode } from "@/hooks/useChatBodyParams";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { getMessageContent } from "@/lib/chat/messages";
import type { MessageReferences } from "@/lib/types/chat";

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

type MessageLike = {
  role?: string;
  parts?: { type?: string; text?: string }[];
};

type ClearSessionChat = <TMessage, TReference, TContext>(helpers: {
  setMessages: (value: TMessage[] | ((prev: TMessage[]) => TMessage[])) => void;
  setReferencesByAssistantIndex: (
    value: TReference[] | ((prev: TReference[]) => TReference[]),
  ) => void;
  setFeedbackSubmitted: (value: boolean | ((prev: boolean) => boolean)) => void;
  setContextUsage: (
    value: TContext | null | ((prev: TContext | null) => TContext | null),
  ) => void;
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
  const [referencesByAssistantIndex, setReferencesByAssistantIndex] = useState<
    (MessageReferences | null)[]
  >([]);
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

  const { messages, sendMessage, status, setMessages } = useChat({
    onError: (error) => {
      console.error("Chat error:", error);
      toast.error(String(error.message ?? error));
    },
    onData: (dataPart) => {
      if (
        dataPart.type === "data-references" &&
        dataPart.data &&
        typeof dataPart.data === "object"
      ) {
        const data = dataPart.data as {
          standalone_question?: string;
          citations?: { source: string; page: string }[];
          reranker_docs?: {
            page_content: string;
            metadata: Record<string, unknown>;
          }[];
          context_usage?: ContextUsage;
          mcp_used?: boolean;
          mcp_tools_used?: string[];
          error?: string;
        };

        if (data.context_usage != null) {
          setContextUsage(data.context_usage);
        }

        if (
          data.standalone_question != null ||
          (data.citations?.length ?? 0) > 0 ||
          (data.reranker_docs?.length ?? 0) > 0 ||
          data.mcp_used === true ||
          (Array.isArray(data.mcp_tools_used) && data.mcp_tools_used.length > 0) ||
          (typeof data.error === "string" && data.error.length > 0)
        ) {
          if (typeof data.error === "string" && data.error.length > 0) {
            toast.error(data.error, "Search unavailable");
          }

          setReferencesByAssistantIndex((prev) => [
            ...prev,
            {
              standalone_question: data.standalone_question,
              citations: data.citations ?? [],
              reranker_docs: data.reranker_docs ?? [],
              mcp_used: data.mcp_used === true,
              mcp_tools_used: Array.isArray(data.mcp_tools_used)
                ? data.mcp_tools_used
                : undefined,
              error: typeof data.error === "string" ? data.error : undefined,
            },
          ]);
          setMaxCitationsToShow(10);
        }
      }
    },
  });

  const {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    showOptimisticSuggestion,
  } = useSuggestions({
    messages,
    status,
    sendMessage,
    selectedModel,
    bodyParams,
    setFeedbackSubmitted,
  });

  const chatContainerRef = useScrollToBottom(status, messages);

  const handleSubmit = useCallback(
    (e: React.SyntheticEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!input.trim()) return;

      setFeedbackSubmitted(false);
      sendMessage({ text: input }, { body: bodyParams });
      setInput("");
    },
    [bodyParams, input, sendMessage],
  );

  const handleRetry = useCallback(() => {
    const lastUserMessage = [...messages]
      .reverse()
      .find((message) => message.role === "user");
    if (lastUserMessage == null) return;

    const text = getMessageContent(lastUserMessage);
    if (!text) return;

    setFeedbackSubmitted(false);
    sendMessage({ text }, { body: bodyParams });
  }, [bodyParams, messages, sendMessage]);

  const handleFeedback = useCallback(
    async (stars: number) => {
      if (messages.length < 2) return;
      const lastUser = messages[messages.length - 2] as MessageLike | undefined;
      const lastAssistant = messages[messages.length - 1] as MessageLike | undefined;
      if (lastUser?.role !== "user" || lastAssistant?.role !== "assistant") return;

      const question = getMessageContent(lastUser);
      const answer = getMessageContent(lastAssistant);

      try {
        const res = await fetch("/api/feedback", {
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
      await fetch(`/api/threads/${encodeURIComponent(threadId)}`, {
        method: "DELETE",
      });
    } catch (error) {
      console.error("Thread cleanup failed:", error);
    }

    clearSessionChat({
      setMessages,
      setReferencesByAssistantIndex,
      setFeedbackSubmitted,
      setContextUsage,
    });
    toast.success("Chat cleared");
  }, [clearSessionChat, setMessages, threadId, toast]);

  return {
    input,
    setInput,
    messages,
    status,
    referencesByAssistantIndex,
    maxCitationsToShow,
    chatContainerRef,
    handleSubmit,
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

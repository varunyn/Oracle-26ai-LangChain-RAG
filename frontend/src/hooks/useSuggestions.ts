import { useCallback, useEffect, useRef, useState } from "react";
import { toApiUrl } from "@/lib/api-base";
import { getMessageContent, type SupportedContent } from "@/lib/chat/messages";

type MessageLike = {
  id?: string;
  role?: string;
  content?: SupportedContent;
};

function fetchSuggestions(
  lastMessage: string,
  lastUserMessage: string | null,
  selectedModel: string,
  threadId: string | null,
  signal: AbortSignal,
  onResult: (suggestions: string[]) => void,
  onDone: () => void
): void {
  fetch(toApiUrl("/api/suggestions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      last_message: lastMessage.slice(-4000),
      last_user_message: lastUserMessage?.slice(-2000) ?? undefined,
      model: selectedModel,
      thread_id: threadId || undefined,
    }),
  })
    .then((r) => r.json())
    .then((data: { suggestions?: string[] }) => {
      if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        onResult(data.suggestions);
      }
      onDone();
    })
    .catch(onDone);
}

export function useSuggestions({
  messages,
  status,
  sendMessage,
  selectedModel,
  threadId,
  setFeedbackSubmitted,
}: {
  messages: MessageLike[];
  status: string;
  sendMessage: (text: string) => void;
  selectedModel: string;
  threadId: string | null;
  setFeedbackSubmitted: (v: boolean) => void;
}) {
  const [dynamicSuggestions, setDynamicSuggestions] = useState<string[] | null>(
    null
  );
  const [pendingSuggestion, setPendingSuggestion] = useState<string | null>(
    null
  );
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const lastSuggestionsMessageIdRef = useRef<string | null>(null);
  const nextRequestIdRef = useRef(0);
  const activeRequestRef = useRef<{
    controller: AbortController;
    id: number;
  } | null>(null);

  const invalidateSuggestionsRequest = useCallback(() => {
    activeRequestRef.current?.controller.abort();
    activeRequestRef.current = null;
    setSuggestionsLoading(false);
  }, []);

  const requestSuggestions = useCallback(
    (lastMessage: string, lastUserMessage: string | null) => {
      invalidateSuggestionsRequest();
      const request = {
        controller: new AbortController(),
        id: ++nextRequestIdRef.current,
      };
      activeRequestRef.current = request;
      setDynamicSuggestions(null);
      setSuggestionsLoading(true);

      fetchSuggestions(
        lastMessage,
        lastUserMessage,
        selectedModel,
        threadId,
        request.controller.signal,
        (suggestions) => {
          if (activeRequestRef.current?.id === request.id) {
            setDynamicSuggestions(suggestions);
          }
        },
        () => {
          if (activeRequestRef.current?.id === request.id) {
            activeRequestRef.current = null;
            setSuggestionsLoading(false);
          }
        }
      );
    },
    [invalidateSuggestionsRequest, selectedModel, threadId]
  );

  const handleSuggestionClick = (suggestion: string) => {
    setFeedbackSubmitted(false);
    setPendingSuggestion(suggestion);
    sendMessage(suggestion);
  };

  useEffect(() => {
    if (!pendingSuggestion || messages.length === 0) {
      return;
    }
    const hasMatchingUserMessage = messages.some((message) => {
      if (message?.role !== "user") {
        return false;
      }
      const text = getMessageContent(message);
      return text.trim() === pendingSuggestion.trim();
    });
    if (hasMatchingUserMessage) {
      queueMicrotask(() => setPendingSuggestion(null));
    }
  }, [messages, pendingSuggestion]);

  useEffect(() => {
    if (status === "submitted" || status === "streaming") {
      activeRequestRef.current?.controller.abort();
      activeRequestRef.current = null;
      queueMicrotask(() => {
        setSuggestionsLoading(false);
        setDynamicSuggestions(null);
      });
    } else if (status === "error") {
      queueMicrotask(() => setPendingSuggestion(null));
    }
  }, [status]);

  useEffect(() => {
    if (
      status !== "ready" ||
      messages.length === 0 ||
      !selectedModel ||
      !threadId
    ) {
      return;
    }
    const last = [...messages]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant" &&
          getMessageContent(message).trim().length > 0
      );
    if (!last) {
      return;
    }
    const text = getMessageContent(last).trim();
    const messageKey = `${threadId}:${last.id ?? "no-message-id"}:${text}`;
    if (lastSuggestionsMessageIdRef.current === messageKey) {
      return;
    }
    lastSuggestionsMessageIdRef.current = messageKey;
    const previousUser = [...messages]
      .reverse()
      .find((msg) => msg.role === "user");
    const previousUserText = previousUser
      ? getMessageContent(previousUser).trim()
      : "";
    requestSuggestions(text, previousUserText || null);
  }, [messages, requestSuggestions, selectedModel, status, threadId]);

  const fetchSuggestionsForText = (
    lastMessageText: string,
    lastUserMessage?: string
  ) => {
    if (!(lastMessageText?.trim() && selectedModel)) {
      return;
    }
    requestSuggestions(lastMessageText, (lastUserMessage || "").trim() || null);
  };

  useEffect(() => invalidateSuggestionsRequest, [invalidateSuggestionsRequest]);

  return {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    fetchSuggestionsForText,
  };
}

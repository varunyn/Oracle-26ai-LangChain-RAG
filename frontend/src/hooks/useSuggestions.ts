import { useEffect, useRef, useState } from "react";
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
  onResult: (suggestions: string[]) => void,
  onDone: () => void
): void {
  fetch(toApiUrl("/api/suggestions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      last_message: lastMessage.slice(-4000),
      last_user_message: lastUserMessage?.slice(-2000) ?? undefined,
      model: selectedModel,
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
  setFeedbackSubmitted,
}: {
  messages: MessageLike[];
  status: string;
  sendMessage: (text: string) => void;
  selectedModel: string;
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
    if (status === "error") {
      queueMicrotask(() => setPendingSuggestion(null));
    }
  }, [status]);

  useEffect(() => {
    if (status !== "ready" || messages.length === 0 || !selectedModel) {
      return;
    }
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") {
      return;
    }
    const text = getMessageContent(last);
    if (!text?.trim()) {
      return;
    }
    const messageKey = last.id ?? `${messages.length}:${text}`;
    if (lastSuggestionsMessageIdRef.current === messageKey) {
      return;
    }
    lastSuggestionsMessageIdRef.current = messageKey;
    queueMicrotask(() => setSuggestionsLoading(true));
    const previousUser = [...messages]
      .reverse()
      .find((msg) => msg.role === "user");
    const previousUserText = previousUser
      ? getMessageContent(previousUser).trim()
      : "";
    fetchSuggestions(
      text,
      previousUserText || null,
      selectedModel,
      setDynamicSuggestions,
      () => setSuggestionsLoading(false)
    );
  }, [status, messages, selectedModel]);

  const fetchSuggestionsForText = (
    lastMessageText: string,
    lastUserMessage?: string
  ) => {
    if (!(lastMessageText?.trim() && selectedModel)) {
      return;
    }
    setSuggestionsLoading(true);
    fetchSuggestions(
      lastMessageText,
      (lastUserMessage || "").trim() || null,
      selectedModel,
      setDynamicSuggestions,
      () => setSuggestionsLoading(false)
    );
  };

  return {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    fetchSuggestionsForText,
  };
}

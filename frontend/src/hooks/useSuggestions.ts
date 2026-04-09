import { useEffect, useRef, useState } from "react";
import { getMessageContent } from "@/lib/chat/messages";

type MessageLike = { id?: string; role?: string; parts?: { type?: string; text?: string }[] };

type ChatBodyParams = {
  model: string;
  thread_id?: string;
  session_id?: string;
  collection_name?: string;
  enable_reranker: boolean;
  enable_tracing: boolean;
  mode: string;
};

function fetchSuggestions(
  lastMessage: string,
  selectedModel: string,
  onResult: (suggestions: string[]) => void,
  onDone: () => void,
): void {
  fetch("/api/suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lastMessage: lastMessage.slice(-4000),
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
  bodyParams,
  setFeedbackSubmitted,
}: {
  messages: MessageLike[];
  status: string;
  sendMessage: (opts: { text: string }, opts2: { body: ChatBodyParams }) => void;
  selectedModel: string;
  bodyParams: ChatBodyParams;
  setFeedbackSubmitted: (v: boolean) => void;
}) {
  const [dynamicSuggestions, setDynamicSuggestions] = useState<string[] | null>(null);
  const [pendingSuggestion, setPendingSuggestion] = useState<string | null>(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const lastSuggestionsMessageIdRef = useRef<string | null>(null);

  const handleSuggestionClick = (suggestion: string) => {
    setFeedbackSubmitted(false);
    setPendingSuggestion(suggestion);
    sendMessage({ text: suggestion }, { body: bodyParams });
  };

  useEffect(() => {
    if (!pendingSuggestion || messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last?.role !== "user") return;
    const text = getMessageContent(last as Parameters<typeof getMessageContent>[0]);
    if (text.trim() === pendingSuggestion.trim()) {
      queueMicrotask(() => setPendingSuggestion(null));
    }
  }, [messages, pendingSuggestion]);

  useEffect(() => {
    if (status === "error") {
      queueMicrotask(() => setPendingSuggestion(null));
    }
  }, [status]);

  useEffect(() => {
    if (status !== "ready" || messages.length === 0 || !selectedModel) return;
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return;
    if (lastSuggestionsMessageIdRef.current === last.id) return;
    const text = getMessageContent(last as Parameters<typeof getMessageContent>[0]);
    if (!text?.trim()) return;
    lastSuggestionsMessageIdRef.current = last.id ?? null;
    queueMicrotask(() => setSuggestionsLoading(true));
    fetchSuggestions(text, selectedModel, setDynamicSuggestions, () =>
      setSuggestionsLoading(false),
    );
  }, [status, messages, selectedModel]);

  const showOptimisticSuggestion =
    pendingSuggestion != null &&
    (messages.length === 0 ||
      (() => {
        const last = messages[messages.length - 1];
        if (last?.role !== "user") return true;
        return (
          getMessageContent(last as Parameters<typeof getMessageContent>[0]).trim() !==
          pendingSuggestion.trim()
        );
      })());

  const fetchSuggestionsForText = (lastMessageText: string) => {
    if (!lastMessageText?.trim() || !selectedModel) return;
    setSuggestionsLoading(true);
    fetchSuggestions(lastMessageText, selectedModel, setDynamicSuggestions, () =>
      setSuggestionsLoading(false),
    );
  };

  return {
    dynamicSuggestions,
    pendingSuggestion,
    suggestionsLoading,
    handleSuggestionClick,
    showOptimisticSuggestion,
    fetchSuggestionsForText,
  };
}

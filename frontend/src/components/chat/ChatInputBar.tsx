"use client";

import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { SUGGESTIONS } from "@/constants/chat";

type ChatInputBarProps = {
  input: string;
  setInput: (v: string) => void;
  onSubmit: (e: React.SyntheticEvent<HTMLFormElement>) => void;
  status: string;
  dynamicSuggestions: string[] | null;
  suggestionsLoading: boolean;
  pendingSuggestion: string | null;
  onSuggestionClick: (suggestion: string) => void;
};

export function ChatInputBar({
  input,
  setInput,
  onSubmit,
  status,
  dynamicSuggestions,
  suggestionsLoading,
  pendingSuggestion,
  onSuggestionClick,
}: ChatInputBarProps): React.ReactElement {
  const showSuggestions =
    status !== "submitted" &&
    status !== "streaming" &&
    pendingSuggestion == null &&
    !suggestionsLoading;

  return (
    <div className="shrink-0 border-t border-border bg-card px-4 py-4 shadow-sm sm:px-6 sm:py-5">
      <div className="mx-auto w-full max-w-4xl space-y-3">
        {showSuggestions && (
          <Suggestions
            className="pb-1 pt-0"
            role="navigation"
            aria-label="Suggested questions"
          >
            {(dynamicSuggestions ?? SUGGESTIONS).map((suggestion) => (
              <Suggestion
                key={suggestion}
                suggestion={suggestion}
                onClick={onSuggestionClick}
              />
            ))}
          </Suggestions>
        )}
        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-3 sm:flex-row sm:items-end"
          aria-label="Chat input"
          data-testid="chat-input-form"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents, policies, or Oracle Cloud data"
            className="min-h-12 flex-1 rounded-lg border border-input bg-background px-4 py-3 text-foreground transition-colors placeholder:text-muted-foreground focus:border-transparent focus:outline-none focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={status !== "ready"}
            aria-label="Message"
            data-testid="chat-input"
          />
          <button
            type="submit"
            disabled={status !== "ready" || !input.trim()}
            className="min-h-12 rounded-lg bg-primary px-6 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 sm:self-auto"
            aria-label="Send message"
            data-testid="chat-send"
          >
            Ask
          </button>
        </form>
      </div>
    </div>
  );
}

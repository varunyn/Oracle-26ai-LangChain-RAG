"use client";

/**
 * Streaming indicator component - shows animated "Thinking..." message
 * Extracted from page.tsx (was STREAMING_INDICATOR constant)
 * React best practice: rendering-hoist - hoisted static JSX
 */
export function StreamingIndicator() {
  return (
    <div
      className="flex items-center space-x-2"
      aria-live="polite"
      aria-busy="true"
      role="status"
      data-testid="chat-streaming-indicator"
    >
      <div className="flex space-x-1">
        <div
          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
          style={{ animationDelay: "0s" }}
        />
        <div
          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
          style={{ animationDelay: "0.1s" }}
        />
        <div
          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
          style={{ animationDelay: "0.2s" }}
        />
      </div>
      <span className="text-sm text-muted-foreground">
        Generating a grounded response...
      </span>
    </div>
  );
}

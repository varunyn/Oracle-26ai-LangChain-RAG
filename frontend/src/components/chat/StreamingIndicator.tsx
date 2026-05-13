"use client";

import { LoaderCircle } from "lucide-react";

/**
 * Streaming indicator component - shows assistant progress while a turn is running.
 * Extracted from page.tsx (was STREAMING_INDICATOR constant)
 * React best practice: rendering-hoist - hoisted static JSX
 */
export function StreamingIndicator() {
  return (
    <div
      className="w-full max-w-md rounded-lg border border-border/70 bg-card px-4 py-3 shadow-sm"
      aria-live="polite"
      aria-busy="true"
      role="status"
      data-testid="chat-streaming-indicator"
    >
      <div className="flex items-center gap-3">
        <span className="inline-flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
          <LoaderCircle className="size-4 animate-spin" aria-hidden />
        </span>
        <div className="min-w-0">
          <div className="text-sm font-medium text-foreground">Working on it</div>
          <div className="mt-0.5 text-xs text-muted-foreground">Preparing response</div>
        </div>
      </div>
      <div className="mt-4 space-y-2" aria-hidden>
        <div className="h-2.5 w-11/12 animate-pulse rounded-full bg-muted" />
        <div className="h-2.5 w-4/5 animate-pulse rounded-full bg-muted" />
        <div className="h-2.5 w-2/3 animate-pulse rounded-full bg-muted" />
      </div>
    </div>
  );
}

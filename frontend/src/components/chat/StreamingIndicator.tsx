"use client";

import { LoaderCircle } from "lucide-react";

type StreamingIndicatorProps = {
  status: string;
};

function getStreamingCopy(status: string): { title: string; detail: string } {
  if (status === "submitted") {
    return {
      title: "Opening answer stream",
      detail: "The question was sent. Waiting for the backend to start streaming.",
    };
  }

  return {
    title: "Preparing response",
    detail: "The stream is active. Tool calls and processing steps will appear as they arrive.",
  };
}

export function StreamingIndicator({
  status,
}: StreamingIndicatorProps): React.ReactElement {
  const copy = getStreamingCopy(status);

  return (
    <div
      className="w-full max-w-2xl rounded-lg border border-border/70 bg-muted/35 px-3.5 py-3"
      aria-live="polite"
      aria-busy="true"
      role="status"
      data-testid="chat-streaming-indicator"
      data-stream-state={status}
    >
      <div className="flex min-w-0 items-center gap-3">
        <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-md bg-background text-muted-foreground ring-1 ring-border">
          <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium leading-5 text-foreground">
            {copy.title}
          </div>
          <div className="text-xs leading-5 text-muted-foreground">
            {copy.detail}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] leading-4 text-muted-foreground">
            <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5">
              Thinking
            </span>
            <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5">
              Tool calls
            </span>
            <span className="rounded border border-border/60 bg-background/60 px-1.5 py-0.5">
              Answer
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-end gap-0.5" aria-hidden>
          <span className="h-2 w-1 animate-pulse rounded-full bg-primary/45" />
          <span className="h-3 w-1 animate-pulse rounded-full bg-primary/60 [animation-delay:120ms]" />
          <span className="h-2.5 w-1 animate-pulse rounded-full bg-primary/45 [animation-delay:240ms]" />
        </div>
      </div>
    </div>
  );
}

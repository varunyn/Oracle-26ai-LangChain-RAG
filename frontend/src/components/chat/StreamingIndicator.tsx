"use client";

import { LoaderCircle } from "lucide-react";

type StreamingIndicatorProps = {
  progress?: string;
  status: string;
};

function getStreamingCopy(status: string, progress?: string): string {
  if (progress?.trim()) {
    return progress;
  }
  if (status === "submitted") {
    return "Opening answer stream";
  }

  return "Preparing response";
}

export function StreamingIndicator({
  progress,
  status,
}: StreamingIndicatorProps): React.ReactElement {
  const copy = getStreamingCopy(status, progress);

  return (
    <div
      aria-busy="true"
      aria-live="polite"
      className="inline-flex w-fit max-w-full items-center gap-2 text-muted-foreground text-sm"
      data-stream-state={status}
      data-testid="chat-streaming-indicator"
      role="status"
    >
      <LoaderCircle aria-hidden className="size-4 shrink-0 animate-spin" />
      <span className="truncate">{copy}</span>
    </div>
  );
}

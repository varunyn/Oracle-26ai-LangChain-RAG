"use client";

import { LoaderCircle } from "lucide-react";

type StreamingIndicatorProps = {
  status: string;
};

function getStreamingCopy(status: string): string {
  if (status === "submitted") {
    return "Opening answer stream";
  }

  return "Preparing response";
}

export function StreamingIndicator({
  status,
}: StreamingIndicatorProps): React.ReactElement {
  const copy = getStreamingCopy(status);

  return (
    <div
      className="inline-flex w-fit max-w-full items-center gap-2 text-sm text-muted-foreground"
      aria-live="polite"
      aria-busy="true"
      role="status"
      data-testid="chat-streaming-indicator"
      data-stream-state={status}
    >
      <LoaderCircle className="size-4 shrink-0 animate-spin" aria-hidden />
      <span className="truncate">{copy}</span>
    </div>
  );
}

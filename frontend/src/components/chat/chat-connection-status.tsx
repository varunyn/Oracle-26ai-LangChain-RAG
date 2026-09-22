"use client";

import type { ChatStatus } from "@/hooks/chat/controller-types";
import type { ReconnectState } from "@/providers/langgraph-stream-provider";

interface ChatConnectionStatusProps {
  reconnect: ReconnectState | null;
  status: ChatStatus;
}

function reconnectReason(cause: unknown): string {
  const message = cause instanceof Error ? cause.message.toLowerCase() : "";
  return message.includes("close") || message.includes("closed")
    ? "The server closed the stream."
    : "The connection was interrupted.";
}

export function ChatConnectionStatus({
  status,
  reconnect,
}: ChatConnectionStatusProps): React.ReactElement | null {
  if (status === "idle") {
    return null;
  }
  if (status === "hydrating") {
    return (
      <div
        className="border-border border-b bg-muted/30 px-4 py-2 text-muted-foreground text-sm"
        data-testid="chat-connection-status"
        role="status"
      >
        Loading conversation…
      </div>
    );
  }
  if (status === "running") {
    return null;
  }
  if (status === "reconnecting" && reconnect) {
    return (
      <div
        className="border-warning/30 border-b bg-warning/10 px-4 py-2 text-sm"
        data-testid="chat-connection-status"
        role="status"
      >
        {reconnectReason(reconnect.cause)} Retrying (attempt {reconnect.attempt}
        ) in {Math.ceil(reconnect.delayMs / 1000)}s.
      </div>
    );
  }
  return (
    <div
      className="border-destructive/30 border-b bg-destructive/10 px-4 py-2 text-destructive text-sm"
      data-testid="chat-connection-status"
      role="alert"
    >
      Chat connection failed. Retry your message to continue.
    </div>
  );
}

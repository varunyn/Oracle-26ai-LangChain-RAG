"use client";

import type { RefObject } from "react";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { getMessageContent } from "@/lib/chat/messages";
import type { MessageReferences } from "@/lib/types/chat";
import { StreamingIndicator } from "@/components/chat/StreamingIndicator";
import { ChatMessageItem } from "@/components/chat/ChatMessageItem";

type MessageLike = {
  id?: string;
  role?: string;
  content?: string;
  references?: MessageReferences | null;
};

type ChatMessageListProps = {
  messages: MessageLike[];
  status: string;
  maxCitationsToShow: number;
  chatContainerRef: RefObject<HTMLDivElement | null>;
  onRetry: () => void;
  onRecoverDirect: () => void;
  onRecoverRagOnly: () => void;
  onFeedback: (stars: number, messageIndex: number) => void;
  feedbackSubmittedMessageIndexes: ReadonlySet<number>;
  enableUserFeedback?: boolean;
};

function hasAssistantProgress(message: MessageLike): boolean {
  if (message.role !== "assistant") return false;
  const progressEvents = message.references?.mcp_progress_events;
  return Array.isArray(progressEvents) && progressEvents.length > 0;
}

function hasActiveAssistantOutput(messages: MessageLike[]): boolean {
  const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  if (!lastAssistant) return false;
  const text = getMessageContent(lastAssistant as Parameters<typeof getMessageContent>[0]).trim();
  return text.length > 0 || hasAssistantProgress(lastAssistant);
}

export function ChatMessageList({
  messages,
  status,
  maxCitationsToShow,
  chatContainerRef,
  onRetry,
  onRecoverDirect,
  onRecoverRagOnly,
  onFeedback,
  feedbackSubmittedMessageIndexes,
  enableUserFeedback,
}: ChatMessageListProps): React.ReactElement {
  const isStreamingTurn = status === "submitted" || status === "streaming";
  const showStreamingIndicator = isStreamingTurn && !hasActiveAssistantOutput(messages);
  const showEmptyState = messages.length === 0 && !isStreamingTurn;

  return (
    <div
      ref={chatContainerRef}
      className="mx-auto flex w-full max-w-4xl flex-1 min-h-0 flex-col overflow-y-auto overflow-x-hidden px-4 py-6 sm:px-6 sm:py-7"
      data-testid="chat-message-list"
      data-chat-status={status}
    >
      {showEmptyState ? (
        <div className="flex flex-1 items-center px-2 py-16 sm:px-4">
          <div className="max-w-xl space-y-3">
            <div className="text-foreground text-xl font-medium">
              Ask a question about your documents
            </div>
            <p className="max-w-md text-sm leading-6 text-muted-foreground">
              Get Oracle-powered answers grounded in your collection, with
              citations you can review as you work.
            </p>
          </div>
        </div>
      ) : null}

      <div className="space-y-6">
        {messages.map((message, index) => {
          const textContent = getMessageContent(
            message as Parameters<typeof getMessageContent>[0],
          );
          const isLastMessage = index === messages.length - 1;
          const isStreaming =
            isLastMessage && (status === "submitted" || status === "streaming");
          const toolName = null;
          const displayContent = textContent;
          const messageReferences: MessageReferences | null =
            message.role === "assistant" ? (message.references ?? null) : null;
          const hasLiveProgress =
            Array.isArray(messageReferences?.mcp_progress_events) &&
            messageReferences.mcp_progress_events.length > 0;

          if (!displayContent && !toolName && !hasLiveProgress) return null;

          const showActions =
            message.role === "assistant" && !!displayContent && !isStreaming;

          return (
            <ChatMessageItem
              key={message.id ?? `message-${index}`}
              message={message}
              displayContent={displayContent}
              toolName={toolName}
              isLastMessage={isLastMessage}
              isStreaming={isStreaming}
              showActions={showActions}
              messageReferences={messageReferences}
              maxCitationsToShow={maxCitationsToShow}
              onRetry={onRetry}
              onRecoverDirect={onRecoverDirect}
              onRecoverRagOnly={onRecoverRagOnly}
              onFeedback={(stars) => onFeedback(stars, index)}
              feedbackSubmitted={feedbackSubmittedMessageIndexes.has(index)}
              enableUserFeedback={enableUserFeedback}
            />
          );
        })}
        {showStreamingIndicator ? (
          <Message from="assistant">
            <MessageContent>
              <StreamingIndicator status={status} />
            </MessageContent>
          </Message>
        ) : null}
      </div>
    </div>
  );
}

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
  parts?: { type?: string; text?: string; data?: unknown }[];
};

type ChatMessageListProps = {
  messages: MessageLike[];
  status: string;
  referencesByAssistantIndex: (MessageReferences | null)[];
  maxCitationsToShow: number;
  chatContainerRef: RefObject<HTMLDivElement | null>;
  onRetry: () => void;
  onFeedback: (stars: number) => void;
  feedbackSubmitted: boolean;
  enableUserFeedback?: boolean;
  pendingSuggestion: string | null;
  showOptimisticSuggestion: boolean;
};

export function ChatMessageList({
  messages,
  status,
  referencesByAssistantIndex,
  maxCitationsToShow,
  chatContainerRef,
  onRetry,
  onFeedback,
  feedbackSubmitted,
  enableUserFeedback,
  pendingSuggestion,
  showOptimisticSuggestion,
}: ChatMessageListProps): React.ReactElement {
  return (
    <div
      ref={chatContainerRef}
      className="mx-auto flex w-full max-w-4xl flex-1 min-h-0 flex-col overflow-y-auto overflow-x-hidden px-4 py-6 sm:px-6 sm:py-7"
      data-testid="chat-message-list"
      data-chat-status={status}
    >
      {messages.length === 0 ? (
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
        const toolRegex = /^⚡ Used tool: (.*?)\n\n/;
        const toolMatch = isStreaming ? null : textContent.match(toolRegex);
        const toolName = toolMatch ? toolMatch[1] : null;
        const displayContent = toolMatch ? textContent.replace(toolRegex, "") : textContent;

        if (!displayContent && !toolName) return null;

        const showActions =
          message.role === "assistant" && !!displayContent && !isStreaming;
        const assistantIndex = messages
          .slice(0, index)
          .filter((m) => m.role === "assistant").length;
        const refPart =
          message.role === "assistant" &&
          Array.isArray(
            (message as { parts?: { type?: string; data?: unknown }[] }).parts,
          )
            ? (
                message as { parts: { type?: string; data?: unknown }[] }
              ).parts.find((p) => p?.type === "data-references")
            : null;
        const refFromParts: MessageReferences | null =
          refPart && refPart.data && typeof refPart.data === "object"
            ? ({
                ...(refPart.data as Record<string, unknown>),
                citations: Array.isArray(
                  (refPart.data as MessageReferences).citations,
                )
                  ? (refPart.data as MessageReferences).citations
                  : [],
                reranker_docs: Array.isArray(
                  (refPart.data as MessageReferences).reranker_docs,
                )
                  ? (refPart.data as MessageReferences).reranker_docs
                  : [],
              } as MessageReferences)
            : null;
        const messageReferences: MessageReferences | null =
          message.role === "assistant"
            ? (refFromParts ??
              referencesByAssistantIndex[assistantIndex] ??
              null)
            : null;

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
            onFeedback={onFeedback}
            feedbackSubmitted={feedbackSubmitted}
            enableUserFeedback={enableUserFeedback}
          />
        );
      })}
      </div>

      {status === "submitted" || status === "streaming" ? (
        <Message from="assistant">
          <MessageContent>
            <StreamingIndicator />
          </MessageContent>
        </Message>
      ) : null}

      {showOptimisticSuggestion && pendingSuggestion != null ? (
        <div key="pending-suggestion" className="pt-2">
          <Message from="user">
            <MessageContent>{pendingSuggestion}</MessageContent>
          </Message>
        </div>
      ) : null}
    </div>
  );
}

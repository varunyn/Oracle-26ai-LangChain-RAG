"use client";

import type { AssembledToolCall } from "@langchain/react";
import { useEffect } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";
import { toolCallsForMessage } from "@/hooks/chat/tool-call-mapping";
import { getMessageContent, type SupportedContent } from "@/lib/chat/messages";
import type { MessageReferences } from "@/lib/types/chat";
import { StreamingIndicator } from "@/components/chat/StreamingIndicator";
import { ChatMessageItem } from "@/components/chat/ChatMessageItem";

type MessageLike = {
  id?: string;
  role?: string;
  content?: SupportedContent;
  toolCallIds?: string[];
  references?: MessageReferences | null;
};

type ChatMessageListProps = {
  messages: MessageLike[];
  toolCalls: AssembledToolCall[];
  status: string;
  maxCitationsToShow: number;
  onRetry: () => void;
  onRecoverDirect: () => void;
  onRecoverRagOnly: () => void;
  onFeedback: (stars: number, messageIndex: number) => void;
  feedbackSubmittedMessageIndexes: ReadonlySet<number>;
  enableUserFeedback?: boolean;
};

function hasAssistantToolCalls(
  message: MessageLike,
  toolCalls: readonly AssembledToolCall[],
): boolean {
  if (message.role !== "assistant") return false;
  return toolCallsForMessage(message.toolCallIds, toolCalls).length > 0;
}

function hasActiveAssistantOutput(
  messages: MessageLike[],
  toolCalls: readonly AssembledToolCall[],
): boolean {
  const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  if (!lastAssistant) return false;
  const text = getMessageContent(lastAssistant).trim();
  return text.length > 0 || hasAssistantToolCalls(lastAssistant, toolCalls);
}

export function ChatMessageList({
  messages,
  toolCalls,
  status,
  maxCitationsToShow,
  onRetry,
  onRecoverDirect,
  onRecoverRagOnly,
  onFeedback,
  feedbackSubmittedMessageIndexes,
  enableUserFeedback,
}: ChatMessageListProps): React.ReactElement {
  const isStreamingTurn = status === "submitted" || status === "streaming";
  const showStreamingIndicator =
    isStreamingTurn && !hasActiveAssistantOutput(messages, toolCalls);
  const showEmptyState = messages.length === 0 && !isStreamingTurn;

  useEffect(() => {
    debugChatStage("ChatMessageList.render", {
      status,
      showStreamingIndicator,
      showEmptyState,
      messages: summarizeMessages(messages),
    });
  }, [messages, showEmptyState, showStreamingIndicator, status]);

  return (
    <Conversation
      className="mx-auto flex w-full max-w-4xl flex-1 min-h-0"
      data-testid="chat-message-list"
      data-chat-status={status}
    >
      <ConversationContent className="overflow-x-hidden px-4 py-6 sm:px-6 sm:py-7">
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
            const textContent = getMessageContent(message);
            const isLastMessage = index === messages.length - 1;
            const isStreaming =
              isLastMessage && (status === "submitted" || status === "streaming");
            const displayContent = textContent;
            const messageReferences: MessageReferences | null =
              message.role === "assistant" ? (message.references ?? null) : null;
            const matchedToolCalls =
              message.role === "assistant"
                ? toolCallsForMessage(message.toolCallIds, toolCalls)
                : [];

            if (!displayContent && matchedToolCalls.length === 0 && !messageReferences?.error) {
              return null;
            }

            const showActions =
              message.role === "assistant" && !!displayContent && !isStreaming;

            return (
              <ChatMessageItem
                key={message.id ?? `message-${index}`}
                message={message}
                displayContent={displayContent}
                toolCalls={matchedToolCalls}
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
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}

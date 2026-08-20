"use client";

import { useEffect } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { ChatMessageItem } from "@/components/chat/ChatMessageItem";
import { StreamingIndicator } from "@/components/chat/StreamingIndicator";
import { debugChatStage, summarizeMessages } from "@/hooks/chat/debug";
import {
  type NativeToolCall,
  toolCallsForMessage,
} from "@/hooks/chat/tool-call-mapping";
import { getMessageContent, type SupportedContent } from "@/lib/chat/messages";
import type { MessageReferences } from "@/lib/types/chat";

interface MessageLike {
  content?: SupportedContent;
  id?: string;
  references?: MessageReferences | null;
  role?: string;
  toolCallIds?: string[];
}

interface ChatMessageListProps {
  enableUserFeedback?: boolean;
  feedbackSubmittedMessageIndexes: ReadonlySet<number>;
  maxCitationsToShow: number;
  messages: MessageLike[];
  onFeedback: (stars: number, messageIndex: number) => void;
  onRecoverDirect: () => void;
  onRecoverRagOnly: () => void;
  onRetry: () => void;
  progress?: string;
  status: string;
  toolCalls: NativeToolCall[];
}

function hasAssistantToolCalls(
  message: MessageLike,
  toolCalls: readonly NativeToolCall[]
): boolean {
  if (message.role !== "assistant") {
    return false;
  }
  return toolCallsForMessage(message.toolCallIds, toolCalls).length > 0;
}

function hasActiveAssistantOutput(
  messages: MessageLike[],
  toolCalls: readonly NativeToolCall[]
): boolean {
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.role === "assistant");
  if (!lastAssistant) {
    return false;
  }
  const text = getMessageContent(lastAssistant).trim();
  return text.length > 0 || hasAssistantToolCalls(lastAssistant, toolCalls);
}

export function ChatMessageList({
  messages,
  progress,
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
  "use memo";
  const isStreamingTurn = status === "submitted" || status === "streaming";
  const showStreamingIndicator =
    isStreamingTurn && !hasActiveAssistantOutput(messages, toolCalls);
  const showEmptyState = messages.length === 0 && !isStreamingTurn;

  useEffect(() => {
    debugChatStage("ChatMessageList.render", {
      status,
      showStreamingIndicator,
      showEmptyState,
      toolCallCount: toolCalls.length,
      messages: summarizeMessages(messages),
    });
  }, [messages, showEmptyState, showStreamingIndicator, status, toolCalls]);

  return (
    <Conversation
      className="mx-auto flex min-h-0 w-full max-w-4xl flex-1"
      data-chat-status={status}
      data-testid="chat-message-list"
    >
      <ConversationContent className="px-4 py-6 sm:px-6 sm:py-7">
        {showEmptyState ? (
          <div className="flex flex-1 items-center px-2 py-16 sm:px-4">
            <div className="max-w-xl space-y-3">
              <div className="font-medium text-foreground text-xl">
                Ask a question about your documents
              </div>
              <p className="max-w-md text-muted-foreground text-sm leading-6">
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
              isLastMessage &&
              (status === "submitted" || status === "streaming");
            const displayContent = textContent;
            const messageReferences: MessageReferences | null =
              message.role === "assistant"
                ? (message.references ?? null)
                : null;
            const matchedToolCalls =
              message.role === "assistant"
                ? toolCallsForMessage(message.toolCallIds, toolCalls)
                : [];

            if (
              !displayContent &&
              matchedToolCalls.length === 0 &&
              !messageReferences?.error
            ) {
              return null;
            }

            const showActions =
              message.role === "assistant" && !!displayContent && !isStreaming;

            return (
              <ChatMessageItem
                displayContent={displayContent}
                enableUserFeedback={enableUserFeedback}
                feedbackSubmitted={feedbackSubmittedMessageIndexes.has(index)}
                isLastMessage={isLastMessage}
                isStreaming={isStreaming}
                key={message.id ?? `message-${index}`}
                maxCitationsToShow={maxCitationsToShow}
                message={message}
                messageReferences={messageReferences}
                onFeedback={(stars) => onFeedback(stars, index)}
                onRecoverDirect={onRecoverDirect}
                onRecoverRagOnly={onRecoverRagOnly}
                onRetry={onRetry}
                showActions={showActions}
                toolCalls={matchedToolCalls}
              />
            );
          })}
          {showStreamingIndicator ? (
            <Message from="assistant">
              <MessageContent>
                <StreamingIndicator progress={progress} status={status} />
              </MessageContent>
            </Message>
          ) : null}
        </div>
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}

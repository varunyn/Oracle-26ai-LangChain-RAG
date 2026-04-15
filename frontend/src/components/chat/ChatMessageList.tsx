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
  maxCitationsToShow: number;
  chatContainerRef: RefObject<HTMLDivElement | null>;
  onRetry: () => void;
  onRecoverDirect: () => void;
  onRecoverRagOnly: () => void;
  onFeedback: (stars: number) => void;
  feedbackSubmitted: boolean;
  enableUserFeedback?: boolean;
  pendingSuggestion: string | null;
  showOptimisticSuggestion: boolean;
};

const TOOL_PREFIX_PATTERNS: RegExp[] = [
  /^⚡\s*Used tool:\s*([^\n]+)\n\n?/i,
  /^(?:🔌\s*\n+)?MCP:\s*([^\n]+)\n+/i,
];

function extractToolHeader(text: string): { toolName: string | null; displayContent: string } {
  for (const pattern of TOOL_PREFIX_PATTERNS) {
    const match = text.match(pattern);
    if (!match) continue;
    const rawToolName = (match[1] ?? "").trim();
    const nextContent = text.replace(pattern, "").trimStart();
    return {
      toolName: rawToolName || null,
      displayContent: nextContent,
    };
  }
  return { toolName: null, displayContent: text };
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
        const parsedHeader = isStreaming
          ? { toolName: null, displayContent: textContent }
          : extractToolHeader(textContent);
        const toolName = parsedHeader.toolName;
        const displayContent = parsedHeader.displayContent;

        if (!displayContent && !toolName) return null;

        const showActions =
          message.role === "assistant" && !!displayContent && !isStreaming;
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
        const sourceParts =
          message.role === "assistant" &&
          Array.isArray(
            (message as {
              parts?: {
                type?: string;
                sourceId?: string;
                url?: string;
                title?: string;
              }[];
            }).parts,
          )
            ? (
                message as {
                  parts: {
                    type?: string;
                    sourceId?: string;
                    url?: string;
                    title?: string;
                  }[];
                }
              ).parts
            : [];

        const sourceCitations =
          sourceParts.length > 0
            ? sourceParts
                .filter((p) => p?.type === "source-document" || p?.type === "source-url")
                .map((p) => ({
                  source:
                    typeof p.url === "string" && p.url
                      ? p.url
                      : typeof p.sourceId === "string" && p.sourceId
                        ? p.sourceId
                        : typeof p.title === "string"
                          ? p.title
                          : "",
                  page: "",
                }))
                .filter((c) => c.source.length > 0)
            : [];
        const dedupedSourceCitations = sourceCitations.filter(
          (citation, index, arr) => arr.findIndex((c) => c.source === citation.source) === index,
        );

        const messageReferences: MessageReferences | null =
          message.role === "assistant"
            ? (refFromParts ??
              (dedupedSourceCitations.length > 0
                ? {
                    citations: dedupedSourceCitations,
                    reranker_docs: [],
                  }
                : null))
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
            onRecoverDirect={onRecoverDirect}
            onRecoverRagOnly={onRecoverRagOnly}
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

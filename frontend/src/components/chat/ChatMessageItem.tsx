"use client";

import type { AssembledToolCall } from "@langchain/react";
import type { ComponentPropsWithoutRef } from "react";
import { memo } from "react";
import { CopyIcon, Star } from "lucide-react";
import type { Components } from "streamdown";
import { SourcesStrip } from "@/components/chat/SourcesStrip";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import type { MessageReferences } from "@/lib/types/chat";
import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { toolCallStateForStatus } from "@/hooks/chat/tool-call-mapping";
import { useToast } from "@/components/toaster";

const markdownComponents: Partial<Components> = {
  ul: (props) => {
    const { className, ...restProps } = props as ComponentPropsWithoutRef<"ul">;
    return (
      <ul
        className={["my-3 list-disc pl-6 space-y-1", className]
          .filter(Boolean)
          .join(" ")}
        {...restProps}
      />
    );
  },
  ol: (props) => {
    const { className, ...restProps } = props as ComponentPropsWithoutRef<"ol">;
    return (
      <ol
        className={["my-3 list-decimal pl-6 space-y-1", className]
          .filter(Boolean)
          .join(" ")}
        {...restProps}
      />
    );
  },
  li: (props) => {
    const { className, ...restProps } = props as ComponentPropsWithoutRef<"li">;
    return (
      <li
        className={["pl-1", className].filter(Boolean).join(" ")}
        {...restProps}
      />
    );
  },
};

type MessageLike = {
  id?: string;
  role?: string;
};

type ChatMessageItemProps = {
  message: MessageLike;
  displayContent: string;
  toolCalls: AssembledToolCall[];
  isLastMessage: boolean;
  isStreaming: boolean;
  showActions: boolean;
  messageReferences: MessageReferences | null;
  maxCitationsToShow: number;
  onRetry: () => void;
  onRecoverDirect: () => void;
  onRecoverRagOnly: () => void;
  onFeedback: (stars: number) => void;
  feedbackSubmitted: boolean;
  enableUserFeedback?: boolean;
};

function ChatMessageItemInner({
  message,
  displayContent,
  toolCalls,
  isLastMessage,
  isStreaming,
  showActions,
  messageReferences,
  maxCitationsToShow,
  onRetry,
  onRecoverDirect,
  onRecoverRagOnly,
  onFeedback,
  feedbackSubmitted,
  enableUserFeedback,
}: ChatMessageItemProps): React.ReactElement {
  const { toast } = useToast();
  const showToolCards = toolCalls.length > 0;
  const hasRefs =
    message.role === "assistant" &&
    !isStreaming &&
    messageReferences?.citations &&
    messageReferences.citations.length > 0;

  const renderContent = () => {
    if (!displayContent) return null;
    if (isStreaming) {
      return (
        <MessageResponse
          components={markdownComponents}
          isAnimating
          mode="streaming"
        >
          {displayContent}
        </MessageResponse>
      );
    }
    if (hasRefs && messageReferences) {
      return (
        <>
          <MessageResponse
            components={markdownComponents}
            isAnimating={isStreaming}
            mode={isStreaming ? "streaming" : "static"}
          >
            {displayContent}
          </MessageResponse>
          <SourcesStrip
            citations={messageReferences.citations}
            maxToShow={maxCitationsToShow}
          />
        </>
      );
    }
    return (
      <MessageResponse
        components={markdownComponents}
        isAnimating={isStreaming}
        mode={isStreaming ? "streaming" : "static"}
      >
        {displayContent}
      </MessageResponse>
    );
  };

  return (
    <div
      data-testid="chat-message-item"
      data-message-role={message.role ?? "user"}
      data-streaming={isStreaming ? "true" : "false"}
    >
      <Message
        from={(message.role ?? "user") as "user" | "assistant" | "system"}
      >
        <MessageContent
          className={
            message.role === "assistant"
              ? "overflow-visible bg-card border border-border rounded-lg px-4 py-3 shadow-sm max-w-full"
              : undefined
          }
        >
          {showToolCards ? (
            <div
              className="mb-3 space-y-2"
              data-testid={isStreaming ? "assistant-activity" : undefined}
            >
              {toolCalls.map((toolCall) => {
                const state = toolCallStateForStatus(toolCall.status);
                return (
                  <Tool
                    key={toolCall.callId}
                    defaultOpen
                    state={state}
                    type={`tool-${toolCall.name}`}
                  >
                    <ToolHeader state={state} type={`tool-${toolCall.name}`} />
                    <ToolContent>
                      <ToolInput input={toolCall.input ?? toolCall.args ?? {}} />
                      <ToolOutput
                        output={
                          toolCall.status === "running"
                            ? toolCall.output ?? "Waiting for tool result..."
                            : toolCall.output ?? "Completed."
                        }
                        errorText={
                          typeof toolCall.error === "string" ? toolCall.error : undefined
                        }
                      />
                    </ToolContent>
                  </Tool>
                );
              })}
            </div>
          ) : null}
          {messageReferences?.error ? (
            <div className="mb-2 inline-flex items-center px-2 py-1 rounded-md bg-destructive/10 text-destructive text-xs font-medium border border-destructive/20 max-w-full">
              <span className="mr-1" aria-hidden>
                ⚠️
              </span>
              <span className="truncate" title={messageReferences.error}>
                Search unavailable: {messageReferences.error}
              </span>
            </div>
          ) : null}
          {renderContent()}
        </MessageContent>
      </Message>
      {showActions ? (
        <MessageActions>
          <MessageAction
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(displayContent);
                toast.success("Message copied to your clipboard.", "Copied");
              } catch {
                toast.error(
                  "We couldn't copy this message. Try again or copy it manually.",
                  "Copy unavailable",
                );
              }
            }}
            label="Copy"
            tooltip="Copy message"
          >
            <CopyIcon className="size-3" />
          </MessageAction>
        </MessageActions>
      ) : null}
      {isLastMessage &&
      message.role === "assistant" &&
      (displayContent.trim().startsWith("Error:") ||
        Boolean(messageReferences?.error)) ? (
        <div className="mt-2">
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={onRetry}
              className="px-3 py-1.5 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={onRecoverDirect}
              className="px-3 py-1.5 text-sm font-medium rounded-md border border-border bg-background text-foreground hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              Try direct mode
            </button>
            <button
              type="button"
              onClick={onRecoverRagOnly}
              className="px-3 py-1.5 text-sm font-medium rounded-md border border-border bg-background text-foreground hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              Try RAG mode
            </button>
          </div>
        </div>
      ) : null}
      {enableUserFeedback &&
      showActions &&
      message.role === "assistant" &&
      !feedbackSubmitted ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-muted-foreground">Rate:</span>
          {[1, 2, 3, 4, 5].map((rating) => (
            <button
              key={rating}
              type="button"
              onClick={() => onFeedback(rating)}
              className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-warning/10 hover:text-warning-foreground focus:outline-none focus:ring-2 focus:ring-warning/40 focus:ring-offset-2"
              aria-label={`Rate ${rating} star${rating > 1 ? "s" : ""}`}
            >
              <Star className="size-4" />
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export const ChatMessageItem = memo(ChatMessageItemInner);

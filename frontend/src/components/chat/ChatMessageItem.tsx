"use client";

import { CopyIcon, Star } from "lucide-react";
import type { ComponentPropsWithoutRef } from "react";
import { memo } from "react";
import type { Components } from "streamdown";
import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { SourcesStrip } from "@/components/chat/SourcesStrip";
import { useToast } from "@/components/toaster";
import {
  type RenderableToolCall,
  toolCallStateForStatus,
} from "@/hooks/chat/tool-call-mapping";
import type { MessageReferences } from "@/lib/types/chat";

const markdownComponents: Partial<Components> = {
  ul: (props) => {
    const { className, ...restProps } = props as ComponentPropsWithoutRef<"ul">;
    return (
      <ul
        className={["my-3 list-disc space-y-1 pl-6", className]
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
        className={["my-3 list-decimal space-y-1 pl-6", className]
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
  toolCalls: RenderableToolCall[];
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
    if (!displayContent) {
      return null;
    }
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
      data-message-role={message.role ?? "user"}
      data-streaming={isStreaming ? "true" : "false"}
      data-testid="chat-message-item"
    >
      <Message
        from={(message.role ?? "user") as "user" | "assistant" | "system"}
      >
        <MessageContent
          className={
            message.role === "assistant"
              ? "max-w-full overflow-visible rounded-lg border border-border bg-card px-4 py-3 shadow-sm"
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
                    defaultOpen
                    key={toolCall.callId}
                    state={state}
                    type={`tool-${toolCall.name}`}
                  >
                    <ToolHeader state={state} type={`tool-${toolCall.name}`} />
                    <ToolContent>
                      <ToolInput input={toolCall.input} />
                      <ToolOutput
                        errorText={
                          typeof toolCall.error === "string"
                            ? toolCall.error
                            : undefined
                        }
                        output={toolCall.output}
                      />
                    </ToolContent>
                  </Tool>
                );
              })}
            </div>
          ) : null}
          {messageReferences?.error ? (
            <div className="mb-2 inline-flex max-w-full items-center rounded-md border border-destructive/20 bg-destructive/10 px-2 py-1 font-medium text-destructive text-xs">
              <span aria-hidden className="mr-1">
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
            label="Copy"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(displayContent);
                toast.success("Message copied to your clipboard.", "Copied");
              } catch {
                toast.error(
                  "We couldn't copy this message. Try again or copy it manually.",
                  "Copy unavailable"
                );
              }
            }}
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
              className="rounded-md bg-primary px-3 py-1.5 font-medium text-primary-foreground text-sm hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              onClick={onRetry}
              type="button"
            >
              Retry
            </button>
            <button
              className="rounded-md border border-border bg-background px-3 py-1.5 font-medium text-foreground text-sm hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              onClick={onRecoverDirect}
              type="button"
            >
              Try direct mode
            </button>
            <button
              className="rounded-md border border-border bg-background px-3 py-1.5 font-medium text-foreground text-sm hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              onClick={onRecoverRagOnly}
              type="button"
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
          <span className="mr-1 text-muted-foreground text-xs">Rate:</span>
          {[1, 2, 3, 4, 5].map((rating) => (
            <button
              aria-label={`Rate ${rating} star${rating > 1 ? "s" : ""}`}
              className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-warning/10 hover:text-warning-foreground focus:outline-none focus:ring-2 focus:ring-warning/40 focus:ring-offset-2"
              key={rating}
              onClick={() => onFeedback(rating)}
              type="button"
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

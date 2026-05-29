"use client";

import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { Children, memo } from "react";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  CircleDashed,
  CopyIcon,
  LoaderCircle,
  Star,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { Components } from "streamdown";
import {
  InlineCitation,
  InlineCitationCard,
  InlineCitationCardBody,
  InlineCitationCardTrigger,
  InlineCitationCarousel,
  InlineCitationCarouselContent,
  InlineCitationCarouselHeader,
  InlineCitationCarouselIndex,
  InlineCitationCarouselItem,
  InlineCitationCarouselNext,
  InlineCitationCarouselPrev,
  InlineCitationQuote,
  InlineCitationSource,
} from "@/components/ai-elements/inline-citation";
import { SourcesStrip } from "@/components/chat/SourcesStrip";
import {
  Tool,
  ToolInput,
  ToolOutput,
  ToolStatusBadge,
  type ToolState,
} from "@/components/ai-elements/tool";
import { CITATION_RUN_REGEX } from "@/constants/chat";
import { splitContentByCitations } from "@/lib/chat/citations";
import type {
  McpProgressEvent,
  MessageReferences,
  McpToolInvocation,
} from "@/lib/types/chat";
import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
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
  toolName: string | null;
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

type ToolRunDisplay = {
  key: string;
  toolName: string;
  args?: unknown;
  result?: string | null;
  error?: string | null;
  state: ToolState;
};

type AssistantActivityProps = {
  displayContent: string;
  progressToolRuns: ToolRunDisplay[];
};

function buildProgressToolRuns(events: McpProgressEvent[]): ToolRunDisplay[] {
  const runs = new Map<string, ToolRunDisplay>();
  const order: string[] = [];

  events.forEach((event, index) => {
    const baseKey =
      typeof event.tool_run_id === "string" && event.tool_run_id.trim()
        ? event.tool_run_id
        : `${event.tool_name}-${index}`;
    const existing = runs.get(baseKey);
    const state: ToolState =
      event.phase === "error"
        ? "output-error"
        : event.phase === "end"
          ? "output-available"
          : "input-available";
    const next: ToolRunDisplay = {
      key: baseKey,
      toolName: event.tool_name,
      args: event.args ?? existing?.args,
      result: event.result ?? existing?.result,
      error: event.error ?? existing?.error,
      state,
    };
    if (!existing) order.push(baseKey);
    runs.set(baseKey, next);
  });

  return order.map((key) => runs.get(key)).filter((run): run is ToolRunDisplay => run != null);
}

function getAssistantActivity({
  displayContent,
  progressToolRuns,
}: AssistantActivityProps): {
  title: string;
  detail: string;
  tone: "neutral" | "active" | "done" | "error";
  icon: LucideIcon;
} {
  const activeRun = [...progressToolRuns].reverse().find((run) => run.state === "input-available");
  const erroredRun = [...progressToolRuns].reverse().find((run) => run.state === "output-error");
  const completedRun = [...progressToolRuns]
    .reverse()
    .find((run) => run.state === "output-available");

  if (erroredRun) {
    return {
      title: "Tool returned an error",
      detail: `${erroredRun.toolName} failed. Retrying or preparing an error response.`,
      tone: "error",
      icon: AlertCircle,
    };
  }
  if (activeRun) {
    return {
      title: `Calling ${activeRun.toolName}`,
      detail: "Arguments are ready. Waiting for the tool result.",
      tone: "active",
      icon: Wrench,
    };
  }
  if (displayContent.trim()) {
    return {
      title: "Writing answer",
      detail: "Streaming the response text as it arrives.",
      tone: "active",
      icon: LoaderCircle,
    };
  }
  if (completedRun) {
    return {
      title: "Processing tool result",
      detail: `${completedRun.toolName} finished. Preparing the final answer.`,
      tone: "done",
      icon: CheckCircle2,
    };
  }
  return {
    title: "Thinking",
    detail: "Choosing whether to answer directly or use a tool.",
    tone: "neutral",
    icon: Brain,
  };
}

function AssistantActivity({
  displayContent,
  progressToolRuns,
}: AssistantActivityProps): React.ReactElement {
  const activity = getAssistantActivity({ displayContent, progressToolRuns });
  const Icon = activity.icon;
  const toneClass =
    activity.tone === "error"
      ? "border-destructive/25 bg-destructive/10 text-destructive"
      : activity.tone === "done"
        ? "border-emerald-600/25 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100"
        : activity.tone === "active"
          ? "border-sky-500/25 bg-sky-500/10 text-sky-950 dark:text-sky-100"
          : "border-border bg-muted/25 text-foreground";

  return (
    <div
      className={`mb-2 rounded-lg border px-3 py-2.5 ${toneClass}`}
      data-testid="assistant-activity"
    >
      <div className="flex min-w-0 items-start gap-2.5">
        <span className="mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded-md bg-background/80 text-current ring-1 ring-border/60">
          <Icon
            className={`size-3.5 ${activity.icon === LoaderCircle ? "animate-spin" : ""}`}
            aria-hidden
          />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium leading-5 text-foreground">
            {activity.title}
          </div>
          <div className="text-xs leading-5 text-muted-foreground">
            {activity.detail}
          </div>
          {progressToolRuns.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {progressToolRuns.slice(-4).map((run) => (
                <span
                  key={run.key}
                  className="inline-flex max-w-full items-center gap-1 rounded border border-border/60 bg-background/70 px-1.5 py-0.5 text-[10px] leading-4 text-muted-foreground"
                >
                  {run.state === "output-available" ? (
                    <CheckCircle2 className="size-3 shrink-0 text-emerald-600" aria-hidden />
                  ) : run.state === "output-error" ? (
                    <AlertCircle className="size-3 shrink-0 text-destructive" aria-hidden />
                  ) : (
                    <CircleDashed className="size-3 shrink-0 animate-spin" aria-hidden />
                  )}
                  <span className="truncate font-mono">{run.toolName}</span>
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ChatMessageItemInner({
  message,
  displayContent,
  toolName,
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
  const mcpToolsUsed = messageReferences?.mcp_tools_used ?? [];
  const toolInvocations: McpToolInvocation[] = (
    messageReferences?.mcp_tool_invocations ?? []
  ).filter(
    (inv) =>
      typeof inv.tool_name === "string" && inv.tool_name.trim().length > 0,
  );
  const progressEvents: McpProgressEvent[] = (
    messageReferences?.mcp_progress_events ?? []
  ).filter(
    (evt) =>
      typeof evt.tool_name === "string" &&
      evt.tool_name.trim().length > 0 &&
      (evt.phase === "start" || evt.phase === "end" || evt.phase === "error"),
  );
  const progressToolRuns = buildProgressToolRuns(progressEvents);
  const toolTimeline = [
    ...(toolName ? [toolName] : []),
    ...mcpToolsUsed.filter((tool) => tool !== toolName),
  ];
  const showToolCards =
    progressToolRuns.length > 0 ||
    toolInvocations.length > 0 ||
    (!toolInvocations.length && toolTimeline.length > 0);
  const segments = splitContentByCitations(displayContent);
  const hasCitationMarkers = segments.some((s) => s.type === "citation");
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
    if (hasRefs && hasCitationMarkers && messageReferences) {
      const citations = messageReferences.citations;
      const rerankerDocs = messageReferences.reranker_docs ?? [];
      const citationComponents: Partial<Components> = {
        p: (props) => {
          const { children, ...pProps } =
            props as ComponentPropsWithoutRef<"p">;
          const processedChildren = Children.map(children, (child) => {
            if (typeof child !== "string") return child;
            const parts: (string | ReactNode)[] = [];
            let lastIndex = 0;
            const regex = new RegExp(CITATION_RUN_REGEX);
            for (const match of child.matchAll(regex)) {
              const matchIndex = match.index ?? 0;

              if (matchIndex > lastIndex) {
                parts.push(child.slice(lastIndex, matchIndex));
              }
              const run = match[0];
              const indices = (run.match(/\d+/g) ?? []).map((n) =>
                parseInt(n, 10),
              );
              if (indices.length > 0) {
                const safeIdx = (i: number) =>
                  Math.min(i - 1, Math.max(0, citations.length - 1));
                const firstC = citations[safeIdx(indices[0])];
                const sourceName = firstC?.source?.split("/").pop() ?? "Source";
                const uniqueSources = [
                  ...new Set(
                    indices
                      .map((i) => citations[safeIdx(i)]?.source ?? "")
                      .filter(Boolean),
                  ),
                ];
                const label =
                  indices.length > 1
                    ? `${sourceName} +${indices.length - 1}`
                    : sourceName;
                parts.push(
                  <InlineCitation
                    key={`cite-${matchIndex}`}
                    className="inline-flex shrink-0 align-baseline ml-0.5"
                  >
                    <InlineCitationCard>
                      <InlineCitationCardTrigger
                        sources={uniqueSources}
                        label={label}
                      />
                      <InlineCitationCardBody>
                        <InlineCitationCarousel>
                          <InlineCitationCarouselHeader>
                            <InlineCitationCarouselPrev />
                            <InlineCitationCarouselNext />
                            <InlineCitationCarouselIndex />
                          </InlineCitationCarouselHeader>
                          <InlineCitationCarouselContent>
                            {indices.map((index) => {
                              const c = citations[safeIdx(index)];
                              const reri = Math.min(
                                index - 1,
                                Math.max(0, rerankerDocs.length - 1),
                              );
                              const doc = rerankerDocs[reri];
                              return (
                                <InlineCitationCarouselItem
                                  key={`${index}-${c?.source ?? ""}`}
                                >
                                  <InlineCitationSource
                                    title={
                                      c?.source?.split("/").pop() ?? "Source"
                                    }
                                    url={c?.source}
                                    description={c?.page ?? undefined}
                                  />
                                  {doc?.page_content ? (
                                    <InlineCitationQuote>
                                      {doc.page_content.slice(0, 500)}
                                      {doc.page_content.length > 500 ? "…" : ""}
                                    </InlineCitationQuote>
                                  ) : null}
                                </InlineCitationCarouselItem>
                              );
                            })}
                          </InlineCitationCarouselContent>
                        </InlineCitationCarousel>
                      </InlineCitationCardBody>
                    </InlineCitationCard>
                  </InlineCitation>,
                );
              }
              lastIndex = matchIndex + run.length;
            }
            if (lastIndex < child.length) {
              parts.push(child.slice(lastIndex));
            }
            return parts.length > 0 ? parts : child;
          });
          return <p {...pProps}>{processedChildren}</p>;
        },
      };
      return (
        <>
          <MessageResponse
            components={{ ...markdownComponents, ...citationComponents }}
            isAnimating={isStreaming}
            mode={isStreaming ? "streaming" : "static"}
          >
            {displayContent}
          </MessageResponse>
          {messageReferences.citations.length > 0 ? (
            <SourcesStrip
              citations={messageReferences.citations}
              rerankerDocs={messageReferences.reranker_docs}
              maxToShow={maxCitationsToShow}
            />
          ) : null}
        </>
      );
    }
    if (hasRefs && !hasCitationMarkers && messageReferences) {
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
            rerankerDocs={messageReferences.reranker_docs}
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
          {isStreaming ? (
            <AssistantActivity
              displayContent={displayContent}
              progressToolRuns={progressToolRuns}
            />
          ) : null}
          {showToolCards ? (
            <div className="mb-2 space-y-1.5">
              {toolInvocations.length > 0
                ? toolInvocations.map((inv, index) => (
                    <Tool
                      key={`${inv.tool_name}-${index}`}
                      type={inv.tool_name}
                      state="output-available"
                      defaultOpen={index === toolInvocations.length - 1}
                    >
                      <div className="w-full min-w-0">
                        <div className="flex min-w-0 items-start justify-between gap-2">
                          <code className="min-w-0 flex-1 break-words rounded border border-border/60 bg-background/50 px-1.5 py-0.5 font-mono text-[10px] leading-snug text-foreground">
                            <span className="text-muted-foreground">
                              {index + 1}.{" "}
                            </span>
                            {inv.tool_name}
                          </code>
                          <ToolStatusBadge
                            state="output-available"
                            className="mt-0.5 shrink-0"
                          />
                        </div>
                        <details className="group mt-2 w-full min-w-0 border-t border-border/50 pt-2">
                          <summary className="cursor-pointer list-none text-[10px] font-medium text-muted-foreground marker:hidden hover:text-foreground">
                            <span className="group-open:hidden">
                              Input & output
                            </span>
                            <span className="hidden group-open:inline">
                              Hide input & output
                            </span>
                          </summary>
                          <div className="mt-2 space-y-2">
                            <div>
                              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                                Input
                              </div>
                              <ToolInput input={inv.args ?? {}} />
                            </div>
                            <div>
                              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                                Output
                              </div>
                              <ToolOutput
                                output={
                                  inv.result ?? "(no tool output in response)"
                                }
                              />
                            </div>
                          </div>
                        </details>
                      </div>
                    </Tool>
                  ))
                : progressToolRuns.length > 0
                  ? progressToolRuns.map((run, index) => {
                      return (
                        <Tool
                          key={run.key}
                          type={run.toolName}
                          state={run.state}
                          defaultOpen={index === progressToolRuns.length - 1}
                        >
                          <div className="w-full min-w-0">
                            <div className="flex min-w-0 items-start justify-between gap-2">
                              <code className="min-w-0 flex-1 break-words rounded border border-border/60 bg-background/50 px-1.5 py-0.5 font-mono text-[10px] leading-snug text-foreground">
                                <span className="text-muted-foreground">
                                  {index + 1}.{" "}
                                </span>
                                {run.toolName}
                              </code>
                              <ToolStatusBadge
                                state={run.state}
                                className="mt-0.5 shrink-0"
                              />
                            </div>
                            <details className="group mt-2 w-full min-w-0 border-t border-border/50 pt-2">
                              <summary className="cursor-pointer list-none text-[10px] font-medium text-muted-foreground marker:hidden hover:text-foreground">
                                <span className="group-open:hidden">
                                  Input & status
                                </span>
                                <span className="hidden group-open:inline">
                                  Hide input & status
                                </span>
                              </summary>
                              <div className="mt-2 space-y-2">
                                <div>
                                  <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                                    Input
                                  </div>
                                  <ToolInput input={run.args ?? {}} />
                                </div>
                                <div>
                                  <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                                    Status
                                  </div>
                                  <ToolOutput
                                    output={
                                      run.state === "input-available"
                                        ? "Waiting for tool result..."
                                        : run.state === "output-error"
                                          ? run.error ?? "Tool execution failed."
                                          : run.result ?? "Completed."
                                    }
                                  />
                                </div>
                              </div>
                            </details>
                          </div>
                        </Tool>
                      );
                    })
                : toolTimeline.map((tool, index) => (
                    <div
                      key={`${tool}-${index}`}
                      className="rounded-md border border-border/50 bg-muted/25 px-2 py-1.5 text-[10px] text-muted-foreground"
                    >
                      <span className="font-medium text-foreground/85">
                        Tool:{" "}
                      </span>
                      <span className="font-mono text-foreground/80">
                        {tool}
                      </span>
                      <span className="mt-1 block leading-snug">
                        Per-call arguments and tool output are shown when the
                        API includes them in the message metadata.
                      </span>
                    </div>
                  ))}
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

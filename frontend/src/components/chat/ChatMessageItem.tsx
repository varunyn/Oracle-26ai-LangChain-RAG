"use client";

import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { memo } from "react";
import {
  AlertCircle,
  CheckCircle2,
  CopyIcon,
  Star,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { Components } from "streamdown";
import { SourcesStrip } from "@/components/chat/SourcesStrip";
import {
  ToolInput,
  ToolOutput,
  type ToolState,
} from "@/components/ai-elements/tool";
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
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

type ToolActivityTimelineProps = {
  isStreaming: boolean;
  progressToolRuns: ToolRunDisplay[];
  toolInvocations: McpToolInvocation[];
  toolTimeline: string[];
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

  return order
    .map((key) => runs.get(key))
    .filter((run): run is ToolRunDisplay => run != null);
}

const TOOL_STATE_TEXT: Record<ToolState, string> = {
  "input-streaming": "Thinking",
  "input-available": "Running",
  "output-available": "Done",
  "output-error": "Error",
};

function toolRunStatus(run: ToolRunDisplay): "complete" | "active" | "pending" {
  return run.state === "input-available" || run.state === "input-streaming"
    ? "active"
    : "complete";
}

function toolRunIcon(run: ToolRunDisplay): LucideIcon {
  if (run.state === "output-error") return AlertCircle;
  if (run.state === "output-available") return CheckCircle2;
  return Wrench;
}

function toolRunDescription(run: ToolRunDisplay): string {
  if (run.state === "output-error") return run.error ?? "Tool execution failed.";
  if (run.state === "output-available") return "Completed.";
  return "Running with prepared arguments.";
}

function toolStepClass(run: ToolRunDisplay): string | undefined {
  if (run.state === "output-error") return "text-destructive";
  if (run.state === "input-available" || run.state === "input-streaming") {
    return "text-sky-950 dark:text-sky-100";
  }
  return undefined;
}

function ToolStepLabel({
  index,
  state,
  toolName,
}: {
  index?: number;
  state: ToolState;
  toolName: string;
}): React.ReactElement {
  return (
    <div className="flex min-w-0 items-center gap-2">
      {index != null ? (
        <span className="shrink-0 text-xs text-muted-foreground">
          {index + 1}.
        </span>
      ) : null}
      <span className="min-w-0 flex-1 truncate font-mono text-xs leading-snug text-foreground">
        {toolName}
      </span>
      <span className="inline-flex h-4 shrink-0 items-center rounded border border-border/60 bg-background/70 px-1 text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
        {TOOL_STATE_TEXT[state]}
      </span>
    </div>
  );
}

function ToolActivitySummary({
  runs,
}: {
  runs: ToolRunDisplay[];
}): React.ReactElement {
  const activeCount = runs.filter(
    (run) => run.state === "input-available" || run.state === "input-streaming",
  ).length;
  const errorCount = runs.filter((run) => run.state === "output-error").length;
  const completeCount = runs.filter(
    (run) => run.state === "output-available",
  ).length;
  const countLabel = (count: number, label: string) =>
    `${count} ${label}${count === 1 ? "" : "s"}`;

  const summary =
    errorCount > 0
      ? countLabel(errorCount, "error")
      : activeCount > 0
        ? countLabel(activeCount, "running")
        : countLabel(completeCount || runs.length, "complete");

  return (
    <span className="flex min-w-0 items-center gap-2">
      <span>Tool activity</span>
      <span className="truncate text-xs text-muted-foreground">{summary}</span>
    </span>
  );
}

function LegacyToolNotice(): React.ReactElement {
  return (
    <div className="text-[10px] leading-snug text-muted-foreground">
      Per-call arguments and output are unavailable for this saved message.
    </div>
  );
}

function ToolRunOutput({
  run,
}: {
  run: ToolRunDisplay;
}): React.ReactElement | null {
  if (
    run.args === undefined &&
    run.result === undefined &&
    run.error === undefined
  ) {
    return <LegacyToolNotice />;
  }

  return (
    <ToolRunDetails
      output={
        run.state === "input-available"
          ? "Waiting for tool result..."
          : run.error ?? run.result ?? "Completed."
      }
      outputLabel={run.state === "output-error" ? "Error" : "Output"}
      toolInput={run.args ?? {}}
    />
  );
}

function ToolRunDetails({
  outputLabel,
  output,
  toolInput,
}: {
  outputLabel: string;
  output: ReactNode;
  toolInput: unknown;
}): React.ReactElement {
  return (
    <details className="group w-full min-w-0 border-t border-border/50 pt-2">
      <summary className="cursor-pointer list-none text-[10px] font-medium text-muted-foreground marker:hidden hover:text-foreground">
        <span className="group-open:hidden">Input & output</span>
        <span className="hidden group-open:inline">Hide input & output</span>
      </summary>
      <div className="mt-2 space-y-2">
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            Input
          </div>
          <ToolInput input={toolInput} />
        </div>
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            {outputLabel}
          </div>
          <ToolOutput output={output} />
        </div>
      </div>
    </details>
  );
}

function ToolActivityTimeline({
  isStreaming,
  progressToolRuns,
  toolInvocations,
  toolTimeline,
}: ToolActivityTimelineProps): React.ReactElement | null {
  const invocationRuns: ToolRunDisplay[] = toolInvocations.map((inv, index) => {
    const hasError =
      typeof inv.error === "string" && inv.error.trim().length > 0;
    return {
      key: `${inv.tool_name}-${index}`,
      toolName: inv.tool_name,
      args: inv.args,
      result: inv.result,
      error: inv.error,
      state: hasError ? "output-error" : "output-available",
    };
  });
  const timelineRuns: ToolRunDisplay[] = toolTimeline.map((tool, index) => ({
    key: `${tool}-${index}`,
    toolName: tool,
    state: "output-available",
  }));
  const runs =
    invocationRuns.length > 0
      ? invocationRuns
      : progressToolRuns.length > 0
        ? progressToolRuns
        : timelineRuns;

  if (runs.length === 0) return null;

  return (
    <ChainOfThought
      className="mb-3 border-y border-border/60 py-2"
      data-testid={isStreaming ? "assistant-activity" : undefined}
      defaultOpen={isStreaming}
    >
      <ChainOfThoughtHeader className="text-xs" icon={Wrench}>
        <ToolActivitySummary runs={runs} />
      </ChainOfThoughtHeader>
      <ChainOfThoughtContent className="space-y-2">
        {runs.map((run, index) => (
          <ChainOfThoughtStep
            className={toolStepClass(run)}
            data-tool-state={run.state}
            data-tool-type={run.toolName}
            description={toolRunDescription(run)}
            icon={toolRunIcon(run)}
            key={run.key}
            label={
              <ToolStepLabel
                index={index}
                state={run.state}
                toolName={run.toolName}
              />
            }
            status={toolRunStatus(run)}
          >
            <ToolRunOutput run={run} />
          </ChainOfThoughtStep>
        ))}
      </ChainOfThoughtContent>
    </ChainOfThought>
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
            <ToolActivityTimeline
              isStreaming={isStreaming}
              progressToolRuns={progressToolRuns}
              toolInvocations={toolInvocations}
              toolTimeline={toolTimeline}
            />
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

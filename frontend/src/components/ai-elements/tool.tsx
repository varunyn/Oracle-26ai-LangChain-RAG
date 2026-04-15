"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

/** Pretty-print JSON strings and objects for tool I/O blocks */
export function formatToolPayload(value: unknown): string {
  if (value === undefined) {
    return "";
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (
      (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
      (trimmed.startsWith("[") && trimmed.endsWith("]"))
    ) {
      try {
        return JSON.stringify(JSON.parse(trimmed), null, 2);
      } catch {
        return value;
      }
    }
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

const toolPreClasses =
  "max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/70 bg-background/80 px-2 py-1.5 text-[10px] leading-relaxed text-foreground/90 [scrollbar-width:thin]";

export type ToolState = "input-available" | "output-available" | "output-error";

const TOOL_STATE_LABEL: Record<ToolState, string> = {
  "input-available": "Pending",
  "output-available": "DONE",
  "output-error": "Error",
};

/** Outer tool call container */
const TOOL_STATE_CLASS: Record<ToolState, string> = {
  "input-available":
    "border-amber-300/50 bg-amber-100/40 text-amber-950 dark:text-amber-100",
  "output-available":
    "border-border bg-muted/35 text-foreground shadow-none dark:bg-muted/25",
  "output-error": "border-destructive/30 bg-destructive/10 text-destructive",
};

/** Compact status pill (keeps success visible without coloring the whole card) */
const TOOL_BADGE_CLASS: Record<ToolState, string> = {
  "input-available":
    "border-amber-400/50 bg-amber-100/70 text-amber-950 dark:text-amber-100",
  "output-available":
    "border-emerald-600/35 bg-emerald-500/12 text-emerald-900 dark:text-emerald-100",
  "output-error": "border-destructive/40 bg-destructive/15 text-destructive",
};

type ToolProps = React.HTMLAttributes<HTMLDivElement> & {
  defaultOpen?: boolean;
  state?: ToolState;
  type?: string;
};

export function Tool({
  className,
  defaultOpen,
  state = "output-available",
  type,
  children,
  ...props
}: ToolProps): React.ReactElement {
  return (
    <div
      className={cn(
        "flex flex-col gap-1.5 rounded-lg border px-2.5 py-2",
        TOOL_STATE_CLASS[state],
        className,
      )}
      data-tool-state={state}
      data-tool-type={type ?? ""}
      data-tool-open={defaultOpen ? "true" : "false"}
      {...props}
    >
      {children}
    </div>
  );
}

export function ToolOutput({
  className,
  errorText,
  output,
  ...props
}: React.HTMLAttributes<HTMLElement> & {
  output: React.ReactNode;
  errorText?: string;
}): React.ReactElement {
  if (errorText) {
    return (
      <div
        className={cn(
          "rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-[11px] leading-5 text-destructive",
          className,
        )}
        {...props}
      >
        {errorText}
      </div>
    );
  }

  if (React.isValidElement(output)) {
    return (
      <div className={cn("overflow-x-auto", className)} {...props}>
        {output}
      </div>
    );
  }

  const text = formatToolPayload(output);

  return (
    <pre
      className={cn(toolPreClasses, className)}
      {...(props as React.HTMLAttributes<HTMLPreElement>)}
    >
      {text}
    </pre>
  );
}

export function ToolInput({
  className,
  input,
  ...props
}: React.HTMLAttributes<HTMLPreElement> & {
  input: unknown;
}): React.ReactElement {
  return (
    <pre className={cn(toolPreClasses, className)} {...props}>
      {formatToolPayload(input)}
    </pre>
  );
}

export function ToolStatusBadge({
  state,
  className,
}: {
  state: ToolState;
  className?: string;
}): React.ReactElement {
  return (
    <span
      className={cn(
        "inline-flex h-4 shrink-0 items-center rounded border px-1 text-[9px] font-semibold uppercase tracking-wide",
        TOOL_BADGE_CLASS[state],
        className,
      )}
    >
      {TOOL_STATE_LABEL[state]}
    </span>
  );
}

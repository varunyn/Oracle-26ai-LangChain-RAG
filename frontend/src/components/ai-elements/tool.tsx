"use client";

import * as React from "react";
import {
  AlertCircle,
  Check,
  ChevronDown,
  CircleDashed,
  Wrench,
} from "lucide-react";
import { useControllableState } from "@radix-ui/react-use-controllable-state";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
  "max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/70 bg-background/80 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-foreground/90 [scrollbar-width:thin]";

export type ToolState =
  | "input-streaming"
  | "input-available"
  | "output-available"
  | "output-error";

const TOOL_STATE_LABEL: Record<ToolState, string> = {
  "input-streaming": "Pending",
  "input-available": "Running",
  "output-available": "Completed",
  "output-error": "Error",
};

/** Outer tool call container */
const TOOL_STATE_CLASS: Record<ToolState, string> = {
  "input-streaming":
    "border-border/80 bg-muted/25 text-foreground shadow-none dark:bg-muted/15",
  "input-available": "border-sky-500/35 bg-sky-500/5 text-foreground",
  "output-available": "border-border/80 bg-card text-foreground shadow-none",
  "output-error": "border-destructive/30 bg-destructive/5 text-destructive",
};

/** Compact status pill (keeps success visible without coloring the whole card) */
const TOOL_BADGE_CLASS: Record<ToolState, string> = {
  "input-streaming": "border-border bg-background/80 text-muted-foreground",
  "input-available":
    "border-sky-500/40 bg-sky-500/10 text-sky-950 dark:text-sky-100",
  "output-available":
    "border-emerald-600/35 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100",
  "output-error": "border-destructive/40 bg-destructive/10 text-destructive",
};

type ToolProps = React.ComponentProps<typeof Collapsible> & {
  state?: ToolState;
  type?: string;
};

function humanizeToolType(type: string | undefined): string {
  if (!type) {
    return "Tool";
  }
  const normalized = type.startsWith("tool-") ? type.slice(5) : type;
  return normalized
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

export function Tool({
  className,
  defaultOpen,
  onOpenChange,
  open: controlledOpen,
  state = "output-available",
  type,
  children,
  ...props
}: ToolProps): React.ReactElement {
  const [open, setOpen] = useControllableState({
    defaultProp: defaultOpen ?? false,
    onChange: onOpenChange,
    prop: controlledOpen,
  });

  React.useEffect(() => {
    if (state === "output-available" || state === "output-error") {
      setOpen(true);
    }
  }, [setOpen, state]);

  return (
    <Collapsible
      className={cn(
        "w-full rounded-xl border px-3 py-2.5",
        TOOL_STATE_CLASS[state],
        className
      )}
      onOpenChange={setOpen}
      open={open}
      data-tool-open={defaultOpen ? "true" : "false"}
      data-tool-state={state}
      data-tool-type={type ?? ""}
      {...props}
    >
      {children}
    </Collapsible>
  );
}

export function ToolHeader({
  className,
  state = "output-available",
  title,
  toolName,
  type,
  ...props
}: Omit<React.ComponentProps<typeof CollapsibleTrigger>, "type"> & {
  state?: ToolState;
  title?: string;
  toolName?: string;
  type?: string;
}): React.ReactElement {
  return (
    <CollapsibleTrigger
      className={cn(
        "group flex min-h-6 w-full items-center justify-between gap-3 text-left",
        className
      )}
      {...props}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden="true"
          className="flex size-5 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground"
        >
          <Wrench className="size-3" strokeWidth={1.8} />
        </span>
        <span className="min-w-0 truncate font-medium text-current text-xs">
          {title ?? toolName ?? humanizeToolType(type)}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <ToolStatusBadge state={state} />
        <ChevronDown
          aria-hidden="true"
          className="size-3.5 text-muted-foreground transition-transform group-data-[state=open]:rotate-180"
        />
      </div>
    </CollapsibleTrigger>
  );
}

export function ToolContent({
  className,
  children,
  ...props
}: React.ComponentProps<typeof CollapsibleContent>): React.ReactElement {
  return (
    <CollapsibleContent
      className={cn(
        "space-y-2.5 overflow-hidden pt-2.5",
        "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-1 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-1",
        className
      )}
      {...props}
    >
      {children}
    </CollapsibleContent>
  );
}

export function ToolOutput({
  className,
  errorText,
  output,
  ...props
}: React.HTMLAttributes<HTMLElement> & {
  output: unknown;
  errorText?: string;
}): React.ReactElement {
  if (errorText) {
    return (
      <div
        className={cn(
          "rounded-md border border-destructive/25 bg-destructive/5 px-2.5 py-2 text-[11px] text-destructive leading-5",
          className
        )}
        data-tool-payload="error"
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
    <div className={cn("space-y-1", className)} {...props}>
      <div className="px-0.5 font-medium text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
        Result
      </div>
      <pre className={toolPreClasses} data-tool-payload="output">
        {text}
      </pre>
    </div>
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
    <div className={cn("space-y-1", className)}>
      <div className="px-0.5 font-medium text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
        Parameters
      </div>
      <pre className={toolPreClasses} data-tool-payload="input" {...props}>
        {formatToolPayload(input)}
      </pre>
    </div>
  );
}

/** @lintignore */
export function ToolStatusBadge({
  state,
  className,
}: {
  state: ToolState;
  className?: string;
}): React.ReactElement {
  return (
    <span
      aria-label={`Tool status: ${TOOL_STATE_LABEL[state]}`}
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1 rounded-md border px-1.5 font-semibold text-[9px] uppercase tracking-[0.08em]",
        TOOL_BADGE_CLASS[state],
        className
      )}
    >
      {state === "input-available" ? (
        <CircleDashed aria-hidden="true" className="size-3" />
      ) : state === "output-error" ? (
        <AlertCircle aria-hidden="true" className="size-3" />
      ) : state === "output-available" ? (
        <Check aria-hidden="true" className="size-3" />
      ) : (
        <CircleDashed aria-hidden="true" className="size-3 animate-spin" />
      )}
      {TOOL_STATE_LABEL[state]}
    </span>
  );
}

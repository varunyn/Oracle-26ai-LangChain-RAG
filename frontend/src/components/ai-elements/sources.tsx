"use client";

import { ChevronDownIcon } from "lucide-react";
import type { ComponentProps } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export type SourcesProps = ComponentProps<typeof Collapsible>;

export function Sources({ className, ...props }: SourcesProps) {
  return (
    <Collapsible
      className={cn("not-prose w-full space-y-2 text-sm", className)}
      {...props}
    />
  );
}

export type SourcesTriggerProps = ComponentProps<typeof CollapsibleTrigger> & {
  count: number;
};

export function SourcesTrigger({
  className,
  count,
  children,
  ...props
}: SourcesTriggerProps) {
  return (
    <CollapsibleTrigger
      className={cn(
        "group inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground",
        className,
      )}
      {...props}
    >
      {children ?? `Used ${count} source${count === 1 ? "" : "s"}`}
      <ChevronDownIcon
        className="size-3.5 transition-transform group-data-[state=open]:rotate-180"
        aria-hidden
      />
    </CollapsibleTrigger>
  );
}

export type SourcesContentProps = ComponentProps<typeof CollapsibleContent>;

export function SourcesContent({
  className,
  ...props
}: SourcesContentProps) {
  return (
    <CollapsibleContent
      className={cn("space-y-1.5 overflow-hidden", className)}
      {...props}
    />
  );
}

export type SourceProps = ComponentProps<"a"> & {
  title: string;
  description?: string;
};

export function Source({
  className,
  description,
  href,
  title,
  ...props
}: SourceProps) {
  const isLink = typeof href === "string" && href.trim().length > 0;

  return (
    <a
      className={cn(
        "block rounded-md border border-border/60 bg-muted/20 px-2.5 py-2 text-xs text-foreground transition-colors",
        isLink && "hover:border-border hover:bg-muted/35",
        !isLink && "pointer-events-none",
        className,
      )}
      href={href}
      rel={isLink ? "noreferrer" : undefined}
      target={isLink ? "_blank" : undefined}
      {...props}
    >
      <span className="block truncate font-medium">{title}</span>
      {description ? (
        <span className="mt-0.5 block truncate text-muted-foreground">
          {description}
        </span>
      ) : null}
    </a>
  );
}

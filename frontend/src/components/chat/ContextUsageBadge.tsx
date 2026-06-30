"use client";

import { useState } from "react";

interface ContextUsage {
  max: number;
  percent: number;
  tokens: number;
}

interface ContextUsageBadgeProps {
  contextUsage: ContextUsage;
}

/**
 * Header badge: shows Context X% by default, tokens/max on hover.
 * Extracted from page.tsx for better code organization.
 */
export function ContextUsageBadge({ contextUsage }: ContextUsageBadgeProps) {
  const [hover, setHover] = useState(false);
  const hasNumbers =
    Number.isFinite(contextUsage.tokens) &&
    Number.isFinite(contextUsage.max) &&
    Number.isFinite(contextUsage.percent);
  if (!hasNumbers) {
    return null;
  }

  return (
    <span
      className="shrink-0 cursor-default rounded border border-border bg-muted/60 px-2 py-1 text-muted-foreground text-xs"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={`Tokens used: ${contextUsage.tokens.toLocaleString()} / ${contextUsage.max.toLocaleString()}`}
    >
      {hover
        ? `${contextUsage.tokens.toLocaleString()} / ${contextUsage.max.toLocaleString()}`
        : `Context: ${contextUsage.percent}%`}
    </span>
  );
}

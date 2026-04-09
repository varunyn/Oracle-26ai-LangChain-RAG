"use client";

import { useState } from "react";

interface ContextUsage {
  tokens: number;
  max: number;
  percent: number;
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
  
  return (
    <span
      className="shrink-0 text-xs text-muted-foreground px-2 py-1 rounded bg-muted/60 border border-border cursor-default"
      title={`Tokens used: ${contextUsage.tokens.toLocaleString()} / ${contextUsage.max.toLocaleString()}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {hover
        ? `${contextUsage.tokens.toLocaleString()} / ${contextUsage.max.toLocaleString()}`
        : `Context: ${contextUsage.percent}%`}
    </span>
  );
}

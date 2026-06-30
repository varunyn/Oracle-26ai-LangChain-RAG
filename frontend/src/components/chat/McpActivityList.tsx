"use client";

import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import type { McpToolActivity } from "@/lib/types/mcp-activity";

function toolState(activity: McpToolActivity): "input-available" | "output-available" | "output-error" {
  if (activity.status === "error") return "output-error";
  if (activity.status === "finished") return "output-available";
  return "input-available";
}

export function McpActivityList({ activities }: { activities: McpToolActivity[] }): React.ReactElement {
  return (
    <div className="space-y-2" data-testid="mcp-tool-activity-list">
      {activities.map((activity) => {
        const state = toolState(activity);
        const toolType =
          activity.serverName != null
            ? `mcp-${activity.serverName}-${activity.toolName}`
            : `mcp-${activity.toolName}`;
        return (
          <Tool key={activity.toolRunId} defaultOpen state={state} type={toolType}>
            <ToolHeader state={state} type={toolType} />
            <ToolContent>
              <div className="text-xs text-muted-foreground">
                {activity.serverName
                  ? `${activity.serverName} / ${activity.toolName}`
                  : activity.toolName}
              </div>
              <ToolInput input={activity.args ?? {}} />
              <ToolOutput
                output={
                  activity.status === "running"
                    ? "Waiting for tool result..."
                    : activity.output ?? "Completed."
                }
                errorText={activity.error ?? undefined}
              />
            </ToolContent>
          </Tool>
        );
      })}
    </div>
  );
}

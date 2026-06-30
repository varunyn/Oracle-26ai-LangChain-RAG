export type McpToolActivityStatus = "running" | "finished" | "error";

export type McpToolActivity = {
  toolRunId: string;
  toolName: string;
  serverName: string | null;
  status: McpToolActivityStatus;
  args: unknown;
  output: unknown;
  error: string | null;
};

type StreamEvent = {
  method?: unknown;
  params?: { data?: unknown };
};

type ActivityPayload = {
  name?: unknown;
  payload?: unknown;
};

function activityFromPayload(payload: unknown): McpToolActivity | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const value = payload as Record<string, unknown>;
  if (
    typeof value.tool_run_id !== "string" ||
    typeof value.tool_name !== "string" ||
    (value.status !== "running" && value.status !== "finished" && value.status !== "error")
  ) {
    return undefined;
  }
  return {
    toolRunId: value.tool_run_id,
    toolName: value.tool_name,
    serverName:
      typeof value.server_name === "string" && value.server_name.trim().length > 0
        ? value.server_name
        : null,
    status: value.status,
    args: value.args ?? null,
    output: value.output ?? null,
    error: typeof value.error === "string" ? value.error : null,
  };
}

export function projectMcpToolActivities(events: readonly unknown[]): McpToolActivity[] {
  const byRunId = new Map<string, McpToolActivity>();
  for (const event of events) {
    if (!event || typeof event !== "object") continue;
    const streamEvent = event as StreamEvent;
    if (streamEvent.method !== "custom") continue;
    const data = streamEvent.params?.data;
    if (!data || typeof data !== "object") continue;
    const activityEvent = data as ActivityPayload;
    if (activityEvent.name !== "mcp_tool_activity") continue;
    const activity = activityFromPayload(activityEvent.payload);
    if (activity) {
      const previous = byRunId.get(activity.toolRunId);
      byRunId.set(activity.toolRunId, {
        ...activity,
        serverName: activity.serverName ?? previous?.serverName ?? null,
        args: activity.args ?? previous?.args ?? null,
        output: activity.output ?? previous?.output ?? null,
        error: activity.error ?? previous?.error ?? null,
      });
    }
  }
  return [...byRunId.values()];
}

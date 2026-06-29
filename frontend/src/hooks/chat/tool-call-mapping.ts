import type { AssembledToolCall } from "@langchain/react";
import type { ToolState } from "@/components/ai-elements/tool";

export function toolCallsForMessage(
  toolCallIds: readonly string[] | undefined,
  toolCalls: readonly AssembledToolCall[],
): AssembledToolCall[] {
  const ids = new Set(toolCallIds ?? []);
  if (ids.size === 0) return [];
  return toolCalls.filter((toolCall) => ids.has(toolCall.callId));
}

export function toolCallStateForStatus(
  status: AssembledToolCall["status"] | undefined,
): ToolState {
  if (status === "error") return "output-error";
  if (status === "finished") return "output-available";
  return "input-available";
}

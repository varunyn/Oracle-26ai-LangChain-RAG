import {
  AIMessage,
  type BaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { AssembledToolCall, ToolCallStatus } from "@langchain/react";
import type { ToolState } from "@/components/ai-elements/tool";
import type { ChatStatus } from "@/hooks/chat/controller-types";
import { getMessageContent } from "@/lib/chat/messages";

/** The product view consumes only the documented assembled projection fields. */
export type NativeToolCall = Pick<
  AssembledToolCall,
  "callId" | "namespace" | "name" | "input" | "output" | "status" | "error"
>;

export interface RenderableToolCall {
  callId: string;
  error: string | undefined;
  input: unknown;
  name: string;
  output: unknown;
  status: ToolCallStatus;
}

/**
 * Seed the root tool projection from the standard messages persisted in a
 * thread checkpoint. @langchain/react currently performs this seeding for
 * scoped projections, but not for the root projection during idle hydration.
 */
export function hydrateToolCallsFromMessages(
  messages: readonly BaseMessage[]
): NativeToolCall[] {
  const calls: NativeToolCall[] = [];
  const callIndexes = new Map<string, number>();

  for (const message of messages) {
    if (AIMessage.isInstance(message)) {
      for (const toolCall of message.tool_calls ?? []) {
        if (
          typeof toolCall.id !== "string" ||
          toolCall.id.length === 0 ||
          typeof toolCall.name !== "string" ||
          toolCall.name.length === 0 ||
          callIndexes.has(toolCall.id)
        ) {
          continue;
        }
        callIndexes.set(toolCall.id, calls.length);
        calls.push({
          callId: toolCall.id,
          namespace: [],
          name: toolCall.name,
          input: toolCall.args,
          output: null,
          status: "running",
          error: undefined,
        });
      }
      continue;
    }

    if (!ToolMessage.isInstance(message)) {
      continue;
    }
    const index = callIndexes.get(message.tool_call_id);
    if (index === undefined) {
      continue;
    }
    const call = calls[index];
    const content = getMessageContent(message);
    calls[index] =
      message.status === "error"
        ? {
            ...call,
            output: null,
            status: "error",
            error: content || "Tool execution failed",
          }
        : {
            ...call,
            output: message.artifact ?? content,
            status: "finished",
            error: undefined,
          };
  }

  return calls;
}

/** Live lifecycle events are newer than checkpoint-derived replay state. */
export function mergeHydratedAndLiveToolCalls(
  hydrated: readonly NativeToolCall[],
  live: readonly NativeToolCall[]
): NativeToolCall[] {
  const merged = new Map(
    hydrated.map((toolCall) => [toolCall.callId, toolCall] as const)
  );
  for (const toolCall of live) {
    merged.set(toolCall.callId, toolCall);
  }
  return [...merged.values()];
}

export function toRenderableToolCall(
  toolCall: NativeToolCall
): RenderableToolCall {
  return {
    callId: toolCall.callId,
    error: toolCall.error,
    input: toolCall.input,
    name: toolCall.name,
    output:
      toolCall.output ??
      (toolCall.status === "running" ? "Waiting for tool result..." : null),
    status: toolCall.status,
  };
}

export function toolCallsForMessage(
  toolCallIds: readonly string[] | undefined,
  toolCalls: readonly NativeToolCall[]
): RenderableToolCall[] {
  if (!toolCallIds?.length) {
    return [];
  }
  const ids = new Set(toolCallIds);
  return toolCalls
    .filter((toolCall) => ids.has(toolCall.callId))
    .map(toRenderableToolCall);
}

export function filterToolCallsForChatStatus(
  toolCalls: readonly NativeToolCall[],
  chatStatus: ChatStatus
): NativeToolCall[] {
  if (chatStatus === "running" || chatStatus === "reconnecting") {
    return [...toolCalls];
  }
  return toolCalls.filter((toolCall) => toolCall.status !== "running");
}

export function toolCallStateForStatus(
  status: RenderableToolCall["status"] | undefined
): ToolState {
  if (status === "error") {
    return "output-error";
  }
  if (status === "finished") {
    return "output-available";
  }
  return "input-available";
}

import type { AssembledToolCall } from "@langchain/react";
import type { ToolState } from "@/components/ai-elements/tool";

export type NativeToolCall = AssembledToolCall;

export type RenderableToolCall = {
  callId: string;
  error?: string;
  input: unknown;
  name: string;
  output: unknown;
  status: "running" | "finished" | "error";
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function nativeToolCallId(toolCall: NativeToolCall): string | undefined {
  if (!isRecord(toolCall)) {
    return;
  }
  const record: Record<string, unknown> = toolCall;
  return (
    stringField(record["callId"]) ??
    stringField(record["id"]) ??
    (isRecord(record["call"]) ? stringField(record["call"]["id"]) : undefined)
  );
}

function normalizeStatus(value: unknown): RenderableToolCall["status"] {
  if (value === "error") {
    return "error";
  }
  if (value === "finished" || value === "completed") {
    return "finished";
  }
  return "running";
}

function resultStatus(
  result: unknown
): RenderableToolCall["status"] | undefined {
  if (!isRecord(result)) {
    return;
  }
  if (result.status === "error") {
    return "error";
  }
  return "finished";
}

function resultContent(result: unknown): unknown {
  return isRecord(result) && "content" in result ? result.content : result;
}

export function toRenderableToolCall(
  toolCall: NativeToolCall
): RenderableToolCall | null {
  if (!isRecord(toolCall)) {
    return null;
  }
  const record: Record<string, unknown> = toolCall;
  const call = isRecord(record["call"]) ? record["call"] : undefined;
  const resultRecord = isRecord(record["result"])
    ? record["result"]
    : undefined;
  const callId = nativeToolCallId(toolCall);
  const name =
    stringField(record["name"]) ??
    stringField(call?.name) ??
    stringField(resultRecord?.name);
  if (!(callId && name)) {
    return null;
  }
  const result = record["result"];
  const status =
    resultStatus(result) ??
    normalizeStatus(record["status"] ?? record["state"]);
  const output = "output" in record ? record["output"] : resultContent(result);
  const error =
    status === "error"
      ? (stringField(record["error"]) ?? JSON.stringify(resultContent(result)))
      : stringField(record["error"]);

  return {
    callId,
    error,
    input: record["input"] ?? record["args"] ?? call?.args ?? {},
    name,
    output:
      status === "running" && output == null
        ? "Waiting for tool result..."
        : (output ?? "Completed."),
    status,
  };
}

export function toolCallsForMessage(
  toolCallIds: readonly string[] | undefined,
  toolCalls: readonly NativeToolCall[]
): RenderableToolCall[] {
  const ids = new Set(toolCallIds ?? []);
  if (ids.size === 0) {
    return [];
  }
  return toolCalls
    .filter((toolCall) => {
      const id = nativeToolCallId(toolCall);
      return id != null && ids.has(id);
    })
    .map(toRenderableToolCall)
    .filter((toolCall): toolCall is RenderableToolCall => toolCall != null);
}

export function filterToolCallsForChatStatus(
  toolCalls: readonly NativeToolCall[],
  chatStatus: string
): NativeToolCall[] {
  if (chatStatus === "submitted" || chatStatus === "streaming") {
    return [...toolCalls];
  }
  return toolCalls.filter((toolCall) => {
    const renderable = toRenderableToolCall(toolCall);
    return renderable != null && renderable.status !== "running";
  });
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

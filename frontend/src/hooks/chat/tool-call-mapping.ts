import type { AssembledToolCall } from "@langchain/react";
import type { BaseMessageWithKwargs } from "@/hooks/chat/references";
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

function isToolMessage(message: BaseMessageWithKwargs): boolean {
  const serialized = message as BaseMessageWithKwargs & {
    type?: unknown;
    role?: unknown;
  };
  const type =
    typeof serialized.type === "string" ? serialized.type.toLowerCase() : "";
  const role =
    typeof serialized.role === "string" ? serialized.role.toLowerCase() : "";
  return type === "tool" || role === "tool";
}

function extractToolCallId(
  message: BaseMessageWithKwargs
): string | undefined {
  const serialized = message as BaseMessageWithKwargs & {
    tool_call_id?: unknown;
  };
  return typeof serialized.tool_call_id === "string" &&
    serialized.tool_call_id.length > 0
    ? serialized.tool_call_id
    : undefined;
}

function extractToolStatus(
  message: BaseMessageWithKwargs
): "error" | "success" | undefined {
  const serialized = message as BaseMessageWithKwargs & {
    status?: unknown;
  };
  if (serialized.status === "error") return "error";
  if (serialized.status === "success") return "success";
  return undefined;
}

function extractContent(value: unknown): unknown {
  if (value == null) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed === ".") return "";
    return value;
  }
  if (Array.isArray(value)) {
    const textParts = value
      .filter(
        (part: unknown): part is { type: string; text: string } =>
          typeof part === "object" &&
          part != null &&
          (part as { type?: string }).type === "text"
      )
      .map((part) => part.text);
    return textParts.length > 0 ? textParts.join("") : String(value);
  }
  return String(value);
}

function extractToolCallsFromContentBlocks(
  content: unknown
): Array<{ id: string; name: string; args: Record<string, unknown> }> {
  if (!Array.isArray(content)) return [];
  return content
    .filter(
      (block): block is Record<string, unknown> =>
        typeof block === "object" &&
        block != null &&
        (block.type === "tool_call" || block.type === "tool_use")
    )
    .map((block) => ({
      id: typeof block.id === "string" ? block.id : "",
      name: typeof block.name === "string" ? block.name : "",
      args:
        block.args && typeof block.args === "object"
          ? (block.args as Record<string, unknown>)
          : block.input && typeof block.input === "object"
            ? (block.input as Record<string, unknown>)
            : {},
    }))
    .filter((tc): tc is { id: string; name: string; args: Record<string, unknown> } => tc.id.length > 0 && tc.name.length > 0);
}

export function deriveToolCallsFromMessages(
  messages: readonly BaseMessageWithKwargs[]
): NativeToolCall[] {
  const toolResultsByCallId = new Map<
    string,
    { output: unknown; status: "error" | "success"; error?: string }
  >();

  for (const message of messages) {
    const raw = message as BaseMessageWithKwargs & {
      type?: unknown;
      tool_calls?: unknown;
    };
    if (!isToolMessage(message)) continue;
    const callId = extractToolCallId(message);
    if (!callId) continue;
    const toolStatus = extractToolStatus(message);
    const content = extractContent(raw.content);
    toolResultsByCallId.set(callId, {
      output: content,
      status: toolStatus === "error" ? "error" : "success",
      error:
        toolStatus === "error"
          ? typeof content === "string"
            ? content
            : "Tool execution failed"
          : undefined,
    });
  }

  const result: NativeToolCall[] = [];

  for (const message of messages) {
    const raw = message as BaseMessageWithKwargs & {
      type?: unknown;
      tool_calls?: Array<{
        id?: string;
        name?: string;
        args?: Record<string, unknown>;
      }>;
    };
    if (raw.type !== "ai") continue;
    let toolCalls = raw.tool_calls;
    if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
      toolCalls = extractToolCallsFromContentBlocks(raw.content) as unknown as Array<{
        id?: string;
        name?: string;
        args?: Record<string, unknown>;
      }>;
      if (toolCalls.length === 0) continue;
    }

    for (const toolCall of toolCalls) {
      const id =
        typeof toolCall.id === "string" && toolCall.id.length > 0
          ? toolCall.id
          : undefined;
      const name =
        typeof toolCall.name === "string" && toolCall.name.length > 0
          ? toolCall.name
          : undefined;
      if (!id || !name) continue;

      const toolResult = toolResultsByCallId.get(id);
      const output = toolResult?.output ?? null;
      const status = toolResult?.status === "error" ? "error" : "finished";
      const error = toolResult?.error;

      result.push({
        name,
        callId: id,
        id,
        namespace: [],
        input: toolCall.args ?? {},
        args: toolCall.args ?? {},
        output,
        status,
        error,
      });
    }
  }

  return result;
}

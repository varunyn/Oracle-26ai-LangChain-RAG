import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import type {
  MessageLike,
  ReferencePayload,
} from "@/hooks/chat/controller-types";
import type { ContextUsage, McpToolInvocation } from "@/lib/types/chat";

export type BaseMessageWithKwargs = BaseMessage & {
  additional_kwargs?: Record<string, unknown>;
  response_metadata?: Record<string, unknown>;
};

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

export function normalizeContextUsage(raw: unknown): ContextUsage | undefined {
  if (!raw || typeof raw !== "object") {
    return;
  }
  const usage = raw as Record<string, unknown>;
  const tokens = toFiniteNumber(usage.tokens);
  const max = toFiniteNumber(usage.max);
  const percent = toFiniteNumber(usage.percent);
  if (tokens == null || max == null || percent == null) {
    return;
  }
  return {
    tokens,
    max,
    percent,
    model_id: typeof usage.model_id === "string" ? usage.model_id : undefined,
  };
}

export function isSameContextUsage(
  a: ContextUsage | null,
  b: ContextUsage
): boolean {
  return (
    a?.tokens === b.tokens &&
    a?.max === b.max &&
    a?.percent === b.percent &&
    a?.model_id === b.model_id
  );
}

export function toRole(message: BaseMessage): "user" | "assistant" | "system" {
  if (HumanMessage.isInstance(message)) {
    return "user";
  }
  if (AIMessage.isInstance(message)) {
    return "assistant";
  }
  if (SystemMessage.isInstance(message)) {
    return "system";
  }

  const serialized = message as BaseMessage & {
    role?: unknown;
    type?: unknown;
  };
  const role =
    typeof serialized.role === "string" ? serialized.role.toLowerCase() : "";
  if (role === "user" || role === "human") {
    return "user";
  }
  if (role === "assistant" || role === "ai") {
    return "assistant";
  }
  if (role === "system") {
    return "system";
  }

  const type =
    typeof serialized.type === "string" ? serialized.type.toLowerCase() : "";
  if (type === "human") {
    return "user";
  }
  if (type === "ai") {
    return "assistant";
  }
  if (type === "system") {
    return "system";
  }
  return "system";
}

/** Extract a displayable error string from a raw error field (string or {type, message}) */
function extractError(raw: Record<string, unknown>): string | undefined {
  const e = raw.error;
  if (typeof e === "string") return e || undefined;
  if (e && typeof e === "object") {
    const obj = e as Record<string, unknown>;
    if (typeof obj.message === "string" && obj.message) return obj.message;
  }
  return undefined;
}

function toReferencePayload(
  raw: Record<string, unknown>
): ReferencePayload | null {
  const hasKnownReferenceField =
    Array.isArray(raw.citations) ||
    Array.isArray(raw.reranker_docs) ||
    Array.isArray(raw.mcp_tools_used) ||
    Array.isArray(raw.mcp_tool_invocations) ||
    typeof raw.trace_id === "string" ||
    typeof raw.standalone_question === "string" ||
    typeof extractError(raw) === "string" ||
    raw.mcp_used === true;
  if (!hasKnownReferenceField) {
    return null;
  }
  return {
    trace_id: typeof raw.trace_id === "string" ? raw.trace_id : undefined,
    standalone_question:
      typeof raw.standalone_question === "string"
        ? raw.standalone_question
        : undefined,
    citations: Array.isArray(raw.citations)
      ? (raw.citations as {
          source: string;
          page: string | null;
          link?: string | null;
        }[])
      : [],
    reranker_docs: Array.isArray(raw.reranker_docs)
      ? (raw.reranker_docs as {
          page_content: string;
          metadata: Record<string, unknown>;
        }[])
      : [],
    context_usage: normalizeContextUsage(raw.context_usage),
    mcp_used: raw.mcp_used === true,
    mcp_tools_used: Array.isArray(raw.mcp_tools_used)
      ? (raw.mcp_tools_used as string[])
      : undefined,
    mcp_tool_invocations: Array.isArray(raw.mcp_tool_invocations)
      ? (raw.mcp_tool_invocations as McpToolInvocation[])
      : undefined,
    error: extractError(raw),
  };
}

export function toReferences(
  message: BaseMessageWithKwargs
): ReferencePayload | null {
  const candidates: unknown[] = [
    message.additional_kwargs,
    message.additional_kwargs?.references,
    message.response_metadata,
  ];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") {
      continue;
    }
    const references = toReferencePayload(candidate as Record<string, unknown>);
    if (references) {
      return references;
    }
  }
  return null;
}

export function traceIdFromMessage(message: MessageLike): string | undefined {
  const traceId = message.references?.trace_id;
  return typeof traceId === "string" && traceId.trim() ? traceId : undefined;
}

export function referencePayloadFromMessage(
  message: MessageLike
): ReferencePayload | null {
  return message.references ?? null;
}

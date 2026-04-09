/**
 * Message-related utilities
 * Extracted from page.tsx for testability and reuse
 */

/** AI SDK message part shape (text part) */
interface MessagePart {
  type?: string;
  text?: string;
}

/** Message with parts array (AI SDK useChat format) */
interface MessageWithParts {
  parts?: MessagePart[];
}

/**
 * Safely extract text content from AI SDK message parts.
 */
export function getMessageContent(message: MessageWithParts | null | undefined): string {
  try {
    if (!message?.parts) return "";
    if (!Array.isArray(message.parts)) return "";
    return message.parts
      .map((part: MessagePart) => {
        if (!part || typeof part !== "object") return "";
        if (part.type === "text") return part.text ?? "";
        return "";
      })
      .join("");
  } catch (e) {
    console.error("Error parsing message content", e);
    return "";
  }
}

/**
 * Generate a unique thread ID for chat sessions.
 * Uses crypto.randomUUID when available, fallback for older environments.
 */
export function generateThreadId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `thread-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

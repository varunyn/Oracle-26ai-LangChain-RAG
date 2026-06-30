/**
 * Message-related utilities
 * Extracted from page.tsx for testability and reuse
 */

export interface TextContentPart {
  text?: string;
  type?: string;
}

export type SupportedContent = string | readonly TextContentPart[];

export interface ChatMessageContent {
  content?: SupportedContent;
}

/**
 * Extract text content from the supported LangChain content shapes used in the app.
 */
export function getMessageContent(
  message: ChatMessageContent | null | undefined
): string {
  if (typeof message?.content === "string") {
    return message.content;
  }
  if (!Array.isArray(message?.content)) {
    return "";
  }
  return message.content
    .filter(
      (part): part is { type: string; text: string } =>
        part?.type === "text" && typeof part.text === "string"
    )
    .map((part) => part.text)
    .join("");
}

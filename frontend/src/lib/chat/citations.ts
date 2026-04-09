/**
 * Citation-related utilities
 * Extracted from page.tsx for testability and reuse
 */

import { CITATION_MARKER_REGEX } from "@/constants/chat";
import type { ContentSegment } from "@/lib/types/chat";

/**
 * Splits content by [1], [2] markers so we can render citations next to the right paragraph.
 */
export function splitContentByCitations(content: string): ContentSegment[] {
  const segments: ContentSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  CITATION_MARKER_REGEX.lastIndex = 0;
  while ((match = CITATION_MARKER_REGEX.exec(content)) !== null) {
    if (match.index > lastIndex) {
      const text = content.slice(lastIndex, match.index);
      if (text.trim().length > 0) segments.push({ type: "text", content: text });
    }
    const num = parseInt(match[1], 10);
    if (num >= 1) segments.push({ type: "citation", index: num });
    lastIndex = CITATION_MARKER_REGEX.lastIndex;
  }
  if (lastIndex < content.length) {
    const text = content.slice(lastIndex);
    if (text.trim().length > 0) segments.push({ type: "text", content: text });
  }
  return segments.length > 0 ? segments : [{ type: "text", content }];
}

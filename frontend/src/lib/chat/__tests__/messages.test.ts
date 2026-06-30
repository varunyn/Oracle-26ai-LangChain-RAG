import { describe, expect, it } from "vitest";

import { getMessageContent } from "../messages";

describe("getMessageContent", () => {
  it("returns string content unchanged", () => {
    const content = getMessageContent({
      content: "Northway terms are net 30.",
    });

    expect(content).toBe("Northway terms are net 30.");
  });

  it("joins structured text blocks and ignores non-text blocks", () => {
    const content = getMessageContent({
      content: [
        { type: "text", text: "**Payment terms**\n" },
        { type: "tool_use", text: "ignored tool text" },
        { type: "text", text: "Northway terms are net 30." },
      ],
    });

    expect(content).toBe("**Payment terms**\nNorthway terms are net 30.");
  });

  it("returns an empty string for missing content", () => {
    expect(getMessageContent(undefined)).toBe("");
    expect(getMessageContent({})).toBe("");
  });

  it("returns an empty string when the content array has no text blocks", () => {
    const content = getMessageContent({
      content: [
        { type: "image_url", text: "ignored" },
        { type: "tool_result" },
      ],
    });

    expect(content).toBe("");
  });
});

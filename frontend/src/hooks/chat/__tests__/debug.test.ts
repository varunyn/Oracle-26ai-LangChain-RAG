import { describe, expect, it } from "vitest";

import { summarizeMessages } from "../debug";

describe("summarizeMessages", () => {
  it("reports duplicate message ids and compact previews", () => {
    const summary = summarizeMessages([
      {
        id: "assistant-1",
        role: "assistant",
        content: "First answer about Summit Technologies policies",
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "First answer about Summit Technologies policies",
      },
      {
        id: "user-1",
        role: "user",
        content: "Tell me about Summit Technologies policies",
      },
    ]);

    expect(summary.total).toBe(3);
    expect(summary.duplicateIds).toEqual(["assistant-1"]);
    expect(summary.messages).toEqual([
      {
        index: 0,
        id: "assistant-1",
        role: "assistant",
        preview: "First answer about Summit Technologies policies",
      },
      {
        index: 1,
        id: "assistant-1",
        role: "assistant",
        preview: "First answer about Summit Technologies policies",
      },
      {
        index: 2,
        id: "user-1",
        role: "user",
        preview: "Tell me about Summit Technologies policies",
      },
    ]);
  });
});

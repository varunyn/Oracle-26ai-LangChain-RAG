import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import { projectStreamMessages } from "../message-projection";

describe("frontend stream message contract", () => {
  it("renders the native stream.messages projection in order", () => {
    const projected = projectStreamMessages([
      new HumanMessage({ id: "user-1", content: "What is Oracle 26ai?" }),
      new AIMessage({
        id: "assistant-1",
        content: "Oracle 26ai is Oracle's latest AI-focused database release.",
        additional_kwargs: {
          citations: [{ source: "guide.pdf", page: "2" }],
        },
      }),
    ]);

    expect(projected).toMatchObject([
      { id: "user-1", role: "user", content: "What is Oracle 26ai?" },
      {
        id: "assistant-1",
        role: "assistant",
        content: "Oracle 26ai is Oracle's latest AI-focused database release.",
        references: { citations: [{ source: "guide.pdf", page: "2" }] },
      },
    ]);
  });

  it("keeps live and replayed native messages as one ordered timeline", () => {
    const user = new HumanMessage({
      id: "user-1",
      content: "What is Oracle 26ai?",
    });
    const assistant = new AIMessage({
      id: "assistant-1",
      content: "Oracle 26ai details.",
    });
    const replay = projectStreamMessages([user, assistant]);

    expect(
      replay.map(({ id, role, content }) => ({ id, role, content }))
    ).toEqual([
      { id: "user-1", role: "user", content: "What is Oracle 26ai?" },
      { id: "assistant-1", role: "assistant", content: "Oracle 26ai details." },
    ]);
  });

  it("maps native assistant tool calls without deriving them from content", () => {
    const projected = projectStreamMessages([
      new AIMessage({
        id: "assistant-1",
        content: "Looking that up now.",
        tool_calls: [
          {
            id: "call-1",
            name: "knowledge_lookup",
            args: { query: "Oracle" },
          },
        ],
      }),
    ]);

    expect(projected).toMatchObject([
      {
        id: "assistant-1",
        role: "assistant",
        content: "Looking that up now.",
        toolCallIds: ["call-1"],
      },
    ]);
  });
});

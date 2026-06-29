import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import { projectStreamMessages } from "../message-projection";

describe("frontend stream message contract", () => {
  it("supports HumanMessage and AIMessage instances from stream.messages", () => {
    const user = new HumanMessage({
      id: "user-1",
      content: "What is Oracle 26ai?",
    });
    const assistant = new AIMessage({
      id: "assistant-1",
      content: [
        { type: "text", text: "Oracle 26ai is Oracle's latest AI-focused database release." },
        { type: "tool_use", text: "ignored tool payload" },
      ],
      additional_kwargs: {
        citations: [{ source: "guide.pdf", page: "2" }],
      },
    });

    const projected = projectStreamMessages({
      streamMessages: [user, assistant],
      liveToolProgressEvents: [],
    });

    expect(projected).toEqual([
      {
        id: "user-1",
        role: "user",
        content: "What is Oracle 26ai?",
        references: null,
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "Oracle 26ai is Oracle's latest AI-focused database release.",
        references: {
          citations: [{ source: "guide.pdf", page: "2" }],
          reranker_docs: [],
          trace_id: undefined,
          standalone_question: undefined,
          context_usage: undefined,
          mcp_used: false,
          mcp_tools_used: undefined,
          mcp_tool_invocations: undefined,
          mcp_progress_events: undefined,
          error: undefined,
        },
      },
    ]);
  });

  it("keeps the latest message when a streamed id is replayed with more complete content", () => {
    const user = new HumanMessage({
      id: "user-1",
      content: "What is Oracle 26ai?",
    });
    const partialAssistant = new AIMessage({
      id: "assistant-1",
      content: [{ type: "text", text: "Oracle 26ai" }],
    });
    const finalAssistant = new AIMessage({
      id: "assistant-1",
      content: [{ type: "text", text: "Oracle 26ai is Oracle's latest AI-focused database release." }],
    });

    const projected = projectStreamMessages({
      streamMessages: [user, partialAssistant, finalAssistant],
      liveToolProgressEvents: [],
    });

    expect(projected.map((message) => message.id)).toEqual(["user-1", "assistant-1"]);
    expect(projected[1]?.content).toBe(
      "Oracle 26ai is Oracle's latest AI-focused database release.",
    );
  });
});

import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import {
  mergeProjectedMessages,
  projectStreamMessages,
  selectMessagesForStatus,
} from "../message-projection";

describe("projectStreamMessages", () => {
  it("drops replayed messages with the same stable id", () => {
    const firstQuestion = new HumanMessage({
      id: "user-1",
      content: "Tell me about Summit Technologies policies",
    });
    const firstAnswer = new AIMessage({
      id: "assistant-1",
      content: "Summit Technologies has net 45 terms.",
    });
    const secondQuestion = new HumanMessage({
      id: "user-2",
      content: "Perform a linear regression using tools",
    });
    const secondAnswer = new AIMessage({
      id: "assistant-2",
      content: "y = 1.54x + 0.44",
    });

    const projected = projectStreamMessages({
      streamMessages: [
        firstQuestion,
        firstAnswer,
        secondQuestion,
        secondAnswer,
        firstAnswer,
        secondAnswer,
      ],
      liveToolProgressEvents: [],
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-1",
      "user-2",
      "assistant-2",
    ]);
  });

  it("keeps the latest copy when the stream replays the same id with updated content", () => {
    const question = new HumanMessage({
      id: "user-1",
      content: "Tell me about Summit Technologies policies",
    });
    const partialAnswer = new AIMessage({
      id: "assistant-1",
      content: "Summit Technologies",
    });
    const finalAnswer = new AIMessage({
      id: "assistant-1",
      content: "Summit Technologies has net 45 terms.",
    });

    const projected = projectStreamMessages({
      streamMessages: [question, partialAnswer, finalAnswer],
      liveToolProgressEvents: [],
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-1",
    ]);
    expect(projected[1]?.content).toBe("Summit Technologies has net 45 terms.");
  });

  it("keeps assistant references attached to the latest message copy", () => {
    const question = new HumanMessage({
      id: "user-1",
      content: "Give me info about payment terms for Northway Solutions",
    });
    const partialAnswer = new AIMessage({
      id: "assistant-final",
      content: "Northway Solutions payment terms",
    });
    const finalAnswer = new AIMessage({
      id: "assistant-final",
      content: "Northway Solutions payment terms are Net 30 days.",
      additional_kwargs: {
        citations: [{ source: "000-Northway_Solutions.pdf", page: "2" }],
        mcp_used: true,
        mcp_tools_used: ["oracle_retrieval"],
      },
    });

    const projected = projectStreamMessages({
      streamMessages: [question, partialAnswer, finalAnswer],
      liveToolProgressEvents: [],
    });

    expect(projected.map((message) => message.id)).toEqual(["user-1", "assistant-final"]);
    expect(projected[1]?.content).toBe("Northway Solutions payment terms are Net 30 days.");
    expect(projected[1]?.references?.citations).toEqual([
      { source: "000-Northway_Solutions.pdf", page: "2" },
    ]);
    expect(projected[1]?.references?.mcp_used).toBe(true);
    expect(projected[1]?.references?.mcp_tools_used).toEqual(["oracle_retrieval"]);
  });

  it("preserves native tool call ids on the owning assistant message", () => {
    const question = new HumanMessage({
      id: "user-1",
      content: "Run linear regression on these points",
    });
    const answer = new AIMessage({
      id: "assistant-1",
      content: [{ type: "text", text: "The best-fit line is y = 1.54x + 0.44." }],
      tool_calls: [
        {
          id: "call-1",
          name: "semantic_search",
          args: { query: "Oracle" },
        },
        {
          id: "call-2",
          name: "list_documents",
          args: {},
        },
      ],
    });

    const projected = projectStreamMessages({
      streamMessages: [question, answer],
      liveToolProgressEvents: [],
    });

    expect(projected).toHaveLength(2);
    expect(projected[1]?.toolCallIds).toEqual(["call-1", "call-2"]);
  });

  it("omits toolCallIds when a message has no valid native tool call ids", () => {
    const assistant = new AIMessage({
      id: "assistant-1",
      content: "No tools here.",
    });

    const projected = projectStreamMessages({
      streamMessages: [assistant],
      liveToolProgressEvents: [],
    });

    expect(projected[0]).not.toHaveProperty("toolCallIds");
  });
});

describe("mergeProjectedMessages", () => {
  it("preserves finalized assistant references from state snapshots", () => {
    const liveMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({
          id: "user-1",
          content: "Give me info about payment terms for Northway Solutions",
        }),
        new AIMessage({
          id: "assistant-1",
          content: "Northway Solutions payment terms are Net 30 days.",
        }),
      ],
    });

    const finalizedMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({
          id: "user-1",
          content: "Give me info about payment terms for Northway Solutions",
        }),
        new AIMessage({
          id: "assistant-1",
          content: "Northway Solutions payment terms are Net 30 days.",
          additional_kwargs: {
            citations: [{ source: "000-Northway_Solutions.pdf", page: "2" }],
          },
        }),
      ],
    });

    expect(mergeProjectedMessages(liveMessages, finalizedMessages)).toMatchObject([
      {
        id: "user-1",
        role: "user",
        content: "Give me info about payment terms for Northway Solutions",
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "Northway Solutions payment terms are Net 30 days.",
        references: {
          citations: [{ source: "000-Northway_Solutions.pdf", page: "2" }],
          reranker_docs: [],
        },
      },
    ]);
  });
});

describe("selectMessagesForStatus", () => {
  it("uses finalized state at completion so citations from state replace live copies", () => {
    const liveMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({ id: "live-answer", content: "Answer without references" }),
      ],
    });
    const finalizedMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({
          id: "final-answer",
          content: "Answer with references",
          additional_kwargs: {
            citations: [{ source: "terms.pdf", page: "2" }],
          },
        }),
      ],
    });

    expect(selectMessagesForStatus(liveMessages, finalizedMessages, "ready")).toEqual(
      finalizedMessages,
    );
  });

  it("keeps the live projection while the run is streaming", () => {
    const liveMessages = [{ id: "live-answer", role: "assistant", content: "Partial" }];
    const finalizedMessages = [
      { id: "final-answer", role: "assistant", content: "Complete" },
    ];

    expect(selectMessagesForStatus(liveMessages, finalizedMessages, "streaming")).toEqual(
      liveMessages,
    );
  });
});

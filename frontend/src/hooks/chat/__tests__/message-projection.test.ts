import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import { projectStreamMessages } from "../message-projection";

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

  it("attaches live MCP progress to the active assistant turn", () => {
    const question = new HumanMessage({
      id: "user-1",
      content: "Run linear regression on these points",
    });
    const answer = new AIMessage({
      id: "assistant-1",
      content: [{ type: "text", text: "The best-fit line is y = 1.54x + 0.44." }],
    });

    const projected = projectStreamMessages({
      streamMessages: [question, answer],
      liveToolProgressEvents: [
        {
          phase: "start",
          tool_name: "Calculator_linear_regression",
          tool_run_id: "tool-1",
          args: { data: [[1, 2], [2, 3.5]] },
        },
      ],
    });

    expect(projected).toHaveLength(2);
    expect(projected[1]?.references?.mcp_progress_events).toEqual([
      {
        phase: "start",
        tool_name: "Calculator_linear_regression",
        tool_run_id: "tool-1",
        args: { data: [[1, 2], [2, 3.5]] },
      },
    ]);
  });
});

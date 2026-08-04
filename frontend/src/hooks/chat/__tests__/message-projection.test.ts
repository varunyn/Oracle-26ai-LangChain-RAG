import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import {
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
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-1",
      "user-2",
      "assistant-2",
    ]);
  });

  it("keeps distinct messages when ids differ even if content matches", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        new HumanMessage({
          id: "optimistic-user",
          content: "Perform a linear regression using tools",
        }),
        new HumanMessage({
          id: "server-user",
          content: "Perform a linear regression using tools",
        }),
      ],
    });

    expect(projected).toEqual([
      {
        id: "optimistic-user",
        role: "user",
        content: "Perform a linear regression using tools",
        references: null,
      },
      {
        id: "server-user",
        role: "user",
        content: "Perform a linear regression using tools",
        references: null,
      },
    ]);
  });

  it("keeps one optimistic user message when an id-less graph copy follows it", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        new HumanMessage({
          id: "optimistic-user",
          content: "Tell me about Oracle 26ai Database.",
        }),
        new HumanMessage({
          content: "Tell me about Oracle 26ai Database.",
        }),
      ],
    });

    expect(projected).toEqual([
      {
        id: "optimistic-user",
        role: "user",
        content: "Tell me about Oracle 26ai Database.",
        references: null,
      },
    ]);
  });

  it("drops an id-less user echo that follows the assistant response", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ content: "Tell me about Oracle 26ai Database." }),
        new AIMessage({ content: "Oracle 26ai deployment details." }),
        new HumanMessage({ content: "Tell me about Oracle 26ai Database." }),
      ],
    });

    expect(projected.map((message) => message.content)).toEqual([
      "Tell me about Oracle 26ai Database.",
      "Oracle 26ai deployment details.",
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
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-final",
    ]);
    expect(projected[1]?.content).toBe(
      "Northway Solutions payment terms are Net 30 days."
    );
    expect(projected[1]?.references?.citations).toEqual([
      { source: "000-Northway_Solutions.pdf", page: "2" },
    ]);
    expect(projected[1]?.references?.mcp_used).toBe(true);
    expect(projected[1]?.references?.mcp_tools_used).toEqual([
      "oracle_retrieval",
    ]);
  });

  it("removes an identical transient assistant projection in favor of the stable answer", () => {
    const user = new HumanMessage({
      id: "user-1",
      content: "What are the payment terms?",
    });
    const stableAnswer = new AIMessage({
      id: "user-1:assistant",
      content: "Payment is due in 30 days.",
      additional_kwargs: {
        citations: [{ source: "terms.pdf", page: null }],
      },
    });
    const transientAnswer = new AIMessage({
      id: "lc_run--model-answer",
      content: "Payment is due in 30 days.",
    });

    const projected = projectStreamMessages({
      streamMessages: [user, stableAnswer, transientAnswer],
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "user-1:assistant",
    ]);
    expect(projected[1]?.references?.citations).toEqual([
      { source: "terms.pdf", page: null },
    ]);
  });

  it("uses native stream.values only after the live RAG stream is ready", () => {
    const user = new HumanMessage({
      id: "user-1",
      content: "What are the payment terms?",
    });
    const streamedModelAnswer = new AIMessage({
      id: "lc_run--model-answer",
      content: "Payment is due in 30 days.",
    });
    const canonicalAnswer = new AIMessage({
      id: "user-1:assistant",
      content: "Payment is due in 30 days.",
      additional_kwargs: {
        mode: "rag",
        citations: [{ source: "terms.pdf", page: null }],
        mcp_used: true,
        mcp_tools_used: ["oracle_retrieval"],
      },
    });

    const live = projectStreamMessages({
      streamMessages: [user, streamedModelAnswer],
    });
    const finalized = projectStreamMessages({
      streamMessages: [user, canonicalAnswer],
    });

    expect(
      selectMessagesForStatus(live, finalized, "streaming").map(
        (message) => message.id
      )
    ).toEqual(["user-1", "lc_run--model-answer"]);
    const projected = selectMessagesForStatus(live, finalized, "ready");
    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "user-1:assistant",
    ]);
    expect(projected[1]?.references?.citations).toEqual([
      { source: "terms.pdf", page: null },
    ]);
    expect(projected[1]?.references?.mcp_used).toBe(true);
    expect(projected[1]?.references?.mcp_tools_used).toEqual([
      "oracle_retrieval",
    ]);
  });

  it("preserves native tool call ids on the owning assistant message", () => {
    const question = new HumanMessage({
      id: "user-1",
      content: "Run linear regression on these points",
    });
    const answer = new AIMessage({
      id: "assistant-1",
      content: [
        { type: "text", text: "The best-fit line is y = 1.54x + 0.44." },
      ],
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
    });

    expect(projected[0]).not.toHaveProperty("toolCallIds");
  });

  it("suppresses placeholder dot content for assistant tool-call messages", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        new AIMessage({
          id: "assistant-tool-call",
          content: ".",
          tool_calls: [
            {
              id: "call-1",
              name: "lookup",
              args: { query: "invoice" },
            },
          ],
        }),
      ],
    });

    expect(projected).toEqual([
      {
        id: "assistant-tool-call",
        role: "assistant",
        content: "",
        toolCallIds: ["call-1"],
        references: null,
      },
    ]);
  });

  it("preserves tool call ids from camelCase stream messages", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        {
          type: "ai",
          id: "assistant-camel",
          content: ".",
          toolCalls: [{ id: "call-1" }, { id: "call-2" }],
        } as never,
      ],
    });

    expect(projected).toEqual([
      {
        id: "assistant-camel",
        role: "assistant",
        content: "",
        toolCallIds: ["call-1", "call-2"],
        references: null,
      },
    ]);
  });

  it("drops raw tool messages from the visible transcript projection", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "Use a tool" }),
        new AIMessage({
          id: "assistant-tool-call",
          content: ".",
          tool_calls: [{ id: "call-1", name: "lookup", args: { q: "x" } }],
        }),
        new ToolMessage({
          id: "tool-1",
          content: "ok",
          tool_call_id: "call-1",
          name: "lookup",
        }),
        new AIMessage({ id: "assistant-final", content: "Done." }),
      ],
    });

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-tool-call",
      "assistant-final",
    ]);
  });
});

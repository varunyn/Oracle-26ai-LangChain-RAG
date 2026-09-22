import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import { projectStreamMessages } from "../message-projection";

describe("projectStreamMessages", () => {
  it("projects the ordered native stream.messages timeline without reconciliation", () => {
    const messages = [
      new HumanMessage({
        id: "user-1",
        content: "Tell me about Summit Technologies policies",
      }),
      new AIMessage({
        id: "assistant-1",
        content: "Summit Technologies has net 45 terms.",
      }),
      new HumanMessage({
        id: "user-2",
        content: "Perform a linear regression using tools",
      }),
      new AIMessage({ id: "assistant-2", content: "y = 1.54x + 0.44" }),
    ];

    expect(projectStreamMessages(messages)).toEqual([
      {
        id: "user-1",
        role: "user",
        content: "Tell me about Summit Technologies policies",
        references: null,
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "Summit Technologies has net 45 terms.",
        references: null,
      },
      {
        id: "user-2",
        role: "user",
        content: "Perform a linear regression using tools",
        references: null,
      },
      {
        id: "assistant-2",
        role: "assistant",
        content: "y = 1.54x + 0.44",
        references: null,
      },
    ]);
  });

  it("preserves distinct native messages with identical content", () => {
    const projected = projectStreamMessages([
      new HumanMessage({ id: "user-1", content: "What is Oracle 26ai?" }),
      new AIMessage({ id: "assistant-1", content: "Oracle 26ai details." }),
      new HumanMessage({ id: "user-2", content: "What is Oracle 26ai?" }),
      new AIMessage({ id: "assistant-2", content: "Oracle 26ai details." }),
    ]);

    expect(
      projected.map(({ id, role, content }) => ({ id, role, content }))
    ).toEqual([
      { id: "user-1", role: "user", content: "What is Oracle 26ai?" },
      { id: "assistant-1", role: "assistant", content: "Oracle 26ai details." },
      { id: "user-2", role: "user", content: "What is Oracle 26ai?" },
      { id: "assistant-2", role: "assistant", content: "Oracle 26ai details." },
    ]);
  });

  it("keeps citations attached to the assistant message that produced them", () => {
    const projected = projectStreamMessages([
      new HumanMessage({
        id: "user-1",
        content: "Give me payment terms for Northway Solutions",
      }),
      new AIMessage({
        id: "assistant-1",
        content: "Northway Solutions payment terms are Net 30 days.",
        additional_kwargs: {
          citations: [{ source: "Northway_Solutions.pdf", page: "2" }],
          mcp_used: true,
          mcp_tools_used: ["oracle_retrieval"],
        },
      }),
    ]);

    expect(projected[1]?.references).toMatchObject({
      citations: [{ source: "Northway_Solutions.pdf", page: "2" }],
      mcp_used: true,
      mcp_tools_used: ["oracle_retrieval"],
    });
  });

  it("preserves native tool call ids on the owning assistant message", () => {
    const projected = projectStreamMessages([
      new HumanMessage({ id: "user-1", content: "Run a regression" }),
      new AIMessage({
        id: "assistant-1",
        content: "The best-fit line is y = 1.54x + 0.44.",
        tool_calls: [
          { id: "call-1", name: "knowledge_lookup", args: { query: "Oracle" } },
          { id: "call-2", name: "list_documents", args: {} },
        ],
      }),
    ]);

    expect(projected[1]?.toolCallIds).toEqual(["call-1", "call-2"]);
  });

  it("does not render raw tool messages in the visible transcript", () => {
    const projected = projectStreamMessages([
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
    ]);

    expect(projected.map((message) => message.id)).toEqual([
      "user-1",
      "assistant-tool-call",
      "assistant-final",
    ]);
  });
});

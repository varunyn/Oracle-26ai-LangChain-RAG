import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import { projectMcpToolActivities } from "@/lib/types/mcp-activity";
import {
  projectStreamMessages,
  selectMessagesForStatus,
} from "../message-projection";

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
      tool_calls: [
        {
          id: "call-1",
          name: "semantic_search",
          args: { query: "Oracle 26ai" },
        },
      ],
      additional_kwargs: {
        citations: [{ source: "guide.pdf", page: "2" }],
      },
    });

    const projected = projectStreamMessages({
      streamMessages: [user, assistant],
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
        toolCallIds: ["call-1"],
        references: {
          citations: [{ source: "guide.pdf", page: "2" }],
          reranker_docs: [],
          trace_id: undefined,
          standalone_question: undefined,
          context_usage: undefined,
          mcp_used: false,
          mcp_tools_used: undefined,
          mcp_tool_invocations: undefined,
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
    });

    expect(projected.map((message) => message.id)).toEqual(["user-1", "assistant-1"]);
    expect(projected[1]?.content).toBe(
      "Oracle 26ai is Oracle's latest AI-focused database release.",
    );
  });

  it("renders citations from serialized LangGraph state messages", () => {
    const projected = projectStreamMessages({
      streamMessages: [
        {
          type: "human",
          id: "user-1",
          content: "What are the payment terms?",
        } as never,
        {
          type: "ai",
          id: "assistant-1",
          content: "Net 30 days.",
          additional_kwargs: {
            citations: [{ source: "terms.pdf", page: null }],
          },
        } as never,
      ],
    });

    expect(projected).toMatchObject([
      { id: "user-1", role: "user" },
      {
        id: "assistant-1",
        role: "assistant",
        references: { citations: [{ source: "terms.pdf", page: null }] },
      },
    ]);
  });

  it("uses finalized state messages at completion instead of maintaining a second fetch-derived message store", () => {
    const liveMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({ id: "assistant-live", content: "Answer without citations" }),
      ],
    });
    const finalizedMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({
          id: "assistant-final",
          content: "Answer with citations",
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

  it("uses finalized state messages for RAG completion without requiring MCP activity", () => {
    const liveMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({ id: "assistant-live", content: "Answer without citations" }),
      ],
    });
    const finalizedMessages = projectStreamMessages({
      streamMessages: [
        new HumanMessage({ id: "user-1", content: "What are the terms?" }),
        new AIMessage({
          id: "assistant-final",
          content: "Answer with citations",
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

  it("keeps MCP activity on the dedicated provider channel adapter instead of mixing it into native tool-call projection", () => {
    const projectedActivities = projectMcpToolActivities([
      {
        method: "custom",
        params: {
          data: {
            name: "mcp_tool_activity",
            payload: {
              tool_run_id: "call-1",
              tool_name: "Calculator_linear_regression",
              status: "running",
              args: { data: [[1, 2], [2, 3.5]] },
            },
          },
        },
      },
      {
        method: "custom",
        params: {
          data: {
            name: "mcp_tool_activity",
            payload: {
              tool_run_id: "call-1",
              tool_name: "Calculator_linear_regression",
              status: "finished",
              output: "{\"slope\":1.54,\"intercept\":0.44}",
            },
          },
        },
      },
    ]);

    expect(projectedActivities).toEqual([
      {
        toolRunId: "call-1",
        toolName: "Calculator_linear_regression",
        serverName: null,
        status: "finished",
        args: { data: [[1, 2], [2, 3.5]] },
        output: "{\"slope\":1.54,\"intercept\":0.44}",
        error: null,
      },
    ]);
  });

  it("keeps MCP activity separate from native tool-call projection", () => {
    const activities = projectMcpToolActivities([
      {
        method: "custom",
        params: {
          data: {
            name: "mcp_tool_activity",
            payload: {
              tool_run_id: "call-1",
              tool_name: "Calculator_linear_regression",
              server_name: "calculator",
              status: "finished",
              output: "ok",
            },
          },
        },
      },
    ]);

    expect(activities[0]).toMatchObject({
      serverName: "calculator",
      toolName: "Calculator_linear_regression",
    });
  });
});

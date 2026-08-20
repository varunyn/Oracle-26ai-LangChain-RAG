import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import {
  deriveToolCallsFromMessages,
  filterToolCallsForChatStatus,
  type NativeToolCall,
  toRenderableToolCall,
  toolCallStateForStatus,
  toolCallsForMessage,
} from "../tool-call-mapping";

describe("filterToolCallsForChatStatus", () => {
  const running: NativeToolCall = {
    callId: "running-1",
    id: "running-1",
    namespace: [],
    name: "WorkflowIntentDecision",
    input: { intent: "calculator" },
    args: { intent: "calculator" },
    output: null,
    status: "running",
    error: undefined,
  };
  const finished: NativeToolCall = {
    callId: "finished-1",
    id: "finished-1",
    namespace: [],
    name: "Calculator_linear_regression",
    input: { data: [[1, 2]] },
    args: { data: [[1, 2]] },
    output: { slope: 1, intercept: 1 },
    status: "finished",
    error: undefined,
  };

  it("keeps running calls while a turn is active", () => {
    expect(filterToolCallsForChatStatus([running], "streaming")).toEqual([
      running,
    ]);
  });

  it("removes unfinished calls after the turn is ready", () => {
    expect(filterToolCallsForChatStatus([running, finished], "ready")).toEqual([
      finished,
    ]);
  });
});

describe("toolCallsForMessage", () => {
  function tc(
    id: string,
    name: string,
    overrides: Partial<NativeToolCall> = {}
  ): NativeToolCall {
    return {
      callId: id,
      id,
      namespace: [],
      name,
      input: {},
      args: {},
      output: null,
      status: "running",
      error: undefined,
      ...overrides,
    };
  }

  it("returns only calls whose callId belongs to the assistant message", () => {
    const toolCalls: NativeToolCall[] = [
      tc("call-1", "semantic_search", { input: { query: "Oracle" }, status: "running" }),
      tc("call-2", "list_documents", { output: ["doc-1"], status: "finished" }),
      tc("call-3", "other_message_tool", { error: "boom", status: "error" }),
    ];

    expect(toolCallsForMessage(["call-1", "call-2"], toolCalls)).toMatchObject([
      { callId: "call-1", status: "running" },
      { callId: "call-2", status: "finished" },
    ]);
  });

  it("normalizes native lifecycle fields for rendering", () => {
    const toolCalls: NativeToolCall[] = [
      tc("call-1", "semantic_search", { input: { query: "Oracle" }, status: "running" }),
      tc("call-2", "list_documents", { output: { documents: ["doc-1"] }, status: "finished" }),
      tc("call-3", "fetch_document", { error: "Document unavailable", status: "error" }),
    ];

    const matched = toolCallsForMessage(
      ["call-1", "call-2", "call-3"],
      toolCalls
    );

    expect(matched).toMatchObject([
      { callId: "call-1", status: "running" },
      { callId: "call-2", status: "finished" },
      { callId: "call-3", status: "error" },
    ]);
    expect(
      matched.map((toolCall) => toolCallStateForStatus(toolCall.status))
    ).toEqual(["input-available", "output-available", "output-error"]);
  });

  it("keeps calls split across assistant messages by native call ids", () => {
    const toolCalls: NativeToolCall[] = [
      tc("call-1", "semantic_search", { input: { query: "Oracle" }, status: "finished" }),
      tc("call-2", "list_documents", { status: "running" }),
    ];

    expect(toolCallsForMessage(["call-1"], toolCalls)).toMatchObject([
      { callId: "call-1" },
    ]);
    expect(toolCallsForMessage(["call-2"], toolCalls)).toMatchObject([
      { callId: "call-2" },
    ]);
    expect(toolCallsForMessage(["missing"], toolCalls)).toEqual([]);
  });

  it("matches assembled tool calls by id alias as well as callId", () => {
    const toolCalls: NativeToolCall[] = [
      tc("call-1", "semantic_search", { input: { query: "Oracle" }, status: "finished" }),
    ];

    expect(toolCallsForMessage(["call-1"], toolCalls)).toMatchObject([
      { callId: "call-1", name: "semantic_search" },
    ]);
  });
});

describe("deriveToolCallsFromMessages", () => {
  it("derives finished tool calls from ai + tool message pairs", () => {
    const messages = [
      new AIMessage({
        id: "ai-1",
        content: ".",
        tool_calls: [
          { id: "call-1", name: "semantic_search", args: { query: "Oracle 26ai" } },
        ],
      }),
      new ToolMessage({
        id: "tool-1",
        content: JSON.stringify({ results: ["doc1", "doc2"] }),
        tool_call_id: "call-1",
        name: "semantic_search",
      }),
    ];

    const derived = deriveToolCallsFromMessages(messages);

    expect(derived).toHaveLength(1);
    const renderable = toRenderableToolCall(derived[0]);
    expect(renderable).toMatchObject({
      callId: "call-1",
      name: "semantic_search",
      status: "finished",
    });
  });

  it("derives error tool calls when tool message has error status", () => {
    const messages = [
      new AIMessage({
        id: "ai-1",
        content: ".",
        tool_calls: [
          { id: "call-1", name: "fetch_document", args: { docId: "123" } },
        ],
      }),
      new ToolMessage({
        id: "tool-1",
        content: "Document not found",
        tool_call_id: "call-1",
        name: "fetch_document",
        status: "error",
      }),
    ];

    const derived = deriveToolCallsFromMessages(messages);

    expect(derived).toHaveLength(1);
    const renderable = toRenderableToolCall(derived[0]);
    expect(renderable).toMatchObject({
      callId: "call-1",
      name: "fetch_document",
      status: "error",
      error: "Document not found",
    });
  });

  it("picks up tool calls from multiple assistant messages", () => {
    const messages = [
      new AIMessage({
        id: "ai-1",
        content: ".",
        tool_calls: [
          { id: "call-1", name: "semantic_search", args: { query: "first" } },
        ],
      }),
      new ToolMessage({
        id: "tool-1",
        content: "first result",
        tool_call_id: "call-1",
      }),
      new AIMessage({
        id: "ai-2",
        content: "The answer is foo.",
      }),
      new AIMessage({
        id: "ai-3",
        content: ".",
        tool_calls: [
          { id: "call-2", name: "list_documents", args: {} },
        ],
      }),
      new ToolMessage({
        id: "tool-2",
        content: "doc-a, doc-b",
        tool_call_id: "call-2",
      }),
      new AIMessage({
        id: "ai-4",
        content: "Final answer.",
      }),
    ];

    const derived = deriveToolCallsFromMessages(messages);

    expect(derived).toHaveLength(2);
    const renderables = derived.map((tc) => toRenderableToolCall(tc));
    expect(renderables[0]).toMatchObject({
      callId: "call-1",
      name: "semantic_search",
      status: "finished",
    });
    expect(renderables[1]).toMatchObject({
      callId: "call-2",
      name: "list_documents",
      status: "finished",
    });
  });

  it("skips ai messages without tool_calls", () => {
    const messages = [
      new AIMessage({ id: "ai-1", content: "Just a thought." }),
    ];

    expect(deriveToolCallsFromMessages(messages)).toEqual([]);
  });

  it("skips tool calls missing id or name", () => {
    const messages = [
      new AIMessage({
        id: "ai-1",
        content: ".",
        tool_calls: [
          { args: { query: "test" } } as never,
        ],
      }),
    ];

    expect(deriveToolCallsFromMessages(messages)).toEqual([]);
  });

  it("produces renderable output that the rest of the pipeline consumes", () => {
    const messages = [
      new AIMessage({
        id: "ai-1",
        content: ".",
        tool_calls: [
          { id: "call-1", name: "semantic_search", args: { query: "test" } },
        ],
      }),
      new ToolMessage({
        id: "tool-1",
        content: JSON.stringify({ title: "Result" }),
        tool_call_id: "call-1",
      }),
    ];

    const derived = deriveToolCallsFromMessages(messages);
    const matched = toolCallsForMessage(["call-1"], derived);

    expect(matched[0]).toMatchObject({
      callId: "call-1",
      name: "semantic_search",
      status: "finished",
    });
  });
});

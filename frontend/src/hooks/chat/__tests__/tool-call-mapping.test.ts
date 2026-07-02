import { describe, expect, it } from "vitest";

import {
  filterToolCallsForChatStatus,
  type NativeToolCall,
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
    status: "running",
  };
  const finished: NativeToolCall = {
    callId: "finished-1",
    id: "finished-1",
    namespace: [],
    name: "Calculator_linear_regression",
    input: { data: [[1, 2]] },
    output: { slope: 1, intercept: 1 },
    status: "finished",
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

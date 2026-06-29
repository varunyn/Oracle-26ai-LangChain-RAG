import type { AssembledToolCall } from "@langchain/react";
import { describe, expect, it } from "vitest";

import { toolCallsForMessage, toolCallStateForStatus } from "../tool-call-mapping";

describe("toolCallsForMessage", () => {
  it("returns only calls whose callId belongs to the assistant message", () => {
    const toolCalls: AssembledToolCall[] = [
      {
        callId: "call-1",
        name: "semantic_search",
        args: { query: "Oracle" },
        input: { query: "Oracle" },
        status: "running",
      },
      {
        callId: "call-2",
        name: "list_documents",
        args: {},
        output: ["doc-1"],
        status: "finished",
      },
      {
        callId: "call-3",
        name: "other_message_tool",
        args: { id: "ignored" },
        error: "boom",
        status: "error",
      },
    ];

    expect(toolCallsForMessage(["call-1", "call-2"], toolCalls)).toEqual([
      toolCalls[0],
      toolCalls[1],
    ]);
  });

  it("preserves native lifecycle fields unchanged", () => {
    const runningCall: AssembledToolCall = {
      callId: "call-1",
      name: "semantic_search",
      args: { query: "Oracle" },
      input: { query: "Oracle" },
      status: "running",
    };
    const finishedCall: AssembledToolCall = {
      callId: "call-2",
      name: "list_documents",
      args: {},
      output: { documents: ["doc-1"] },
      status: "finished",
    };
    const erroredCall: AssembledToolCall = {
      callId: "call-3",
      name: "fetch_document",
      args: { id: "doc-1" },
      error: "Document unavailable",
      status: "error",
    };

    const matched = toolCallsForMessage(
      ["call-1", "call-2", "call-3"],
      [runningCall, finishedCall, erroredCall],
    );

    expect(matched).toEqual([runningCall, finishedCall, erroredCall]);
    expect(matched.map((toolCall) => toolCallStateForStatus(toolCall.status))).toEqual([
      "input-available",
      "output-available",
      "output-error",
    ]);
  });

  it("keeps calls split across assistant messages by native call ids", () => {
    const toolCalls: AssembledToolCall[] = [
      {
        callId: "call-1",
        name: "semantic_search",
        args: { query: "Oracle" },
        status: "finished",
      },
      {
        callId: "call-2",
        name: "list_documents",
        args: {},
        status: "running",
      },
    ];

    expect(toolCallsForMessage(["call-1"], toolCalls)).toEqual([toolCalls[0]]);
    expect(toolCallsForMessage(["call-2"], toolCalls)).toEqual([toolCalls[1]]);
    expect(toolCallsForMessage(["missing"], toolCalls)).toEqual([]);
  });
});

import { describe, expect, it } from "vitest";

import {
  type NativeToolCall,
  toolCallStateForStatus,
  toolCallsForMessage,
} from "../tool-call-mapping";

describe("toolCallsForMessage", () => {
  it("returns only calls whose callId belongs to the assistant message", () => {
    const toolCalls: NativeToolCall[] = [
      {
        callId: "call-1",
        id: "call-1",
        name: "semantic_search",
        args: { query: "Oracle" },
        input: { query: "Oracle" },
        status: "running",
      },
      {
        callId: "call-2",
        id: "call-2",
        name: "list_documents",
        args: {},
        output: ["doc-1"],
        status: "finished",
      },
      {
        callId: "call-3",
        id: "call-3",
        name: "other_message_tool",
        args: { id: "ignored" },
        error: "boom",
        status: "error",
      },
    ];

    expect(toolCallsForMessage(["call-1", "call-2"], toolCalls)).toMatchObject([
      {
        callId: "call-1",
        input: { query: "Oracle" },
        name: "semantic_search",
        status: "running",
      },
      {
        callId: "call-2",
        input: {},
        name: "list_documents",
        output: ["doc-1"],
        status: "finished",
      },
    ]);
  });

  it("normalizes native lifecycle fields for rendering", () => {
    const runningCall: NativeToolCall = {
      callId: "call-1",
      id: "call-1",
      name: "semantic_search",
      args: { query: "Oracle" },
      input: { query: "Oracle" },
      status: "running",
    };
    const finishedCall: NativeToolCall = {
      callId: "call-2",
      id: "call-2",
      name: "list_documents",
      args: {},
      output: { documents: ["doc-1"] },
      status: "finished",
    };
    const erroredCall: NativeToolCall = {
      callId: "call-3",
      id: "call-3",
      name: "fetch_document",
      args: { id: "doc-1" },
      error: "Document unavailable",
      status: "error",
    };

    const matched = toolCallsForMessage(
      ["call-1", "call-2", "call-3"],
      [runningCall, finishedCall, erroredCall]
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
      {
        callId: "call-1",
        id: "call-1",
        name: "semantic_search",
        args: { query: "Oracle" },
        status: "finished",
      },
      {
        callId: "call-2",
        id: "call-2",
        name: "list_documents",
        args: {},
        status: "running",
      },
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
      {
        callId: "call-1",
        id: "call-1",
        name: "semantic_search",
        args: { query: "Oracle" },
        status: "finished",
      },
    ];

    expect(toolCallsForMessage(["call-1"], toolCalls)).toMatchObject([
      { callId: "call-1", name: "semantic_search" },
    ]);
  });

  it("supports ToolCallWithResult objects from stream.toolCalls", () => {
    const toolCalls: NativeToolCall[] = [
      {
        id: "call-1",
        call: {
          id: "call-1",
          name: "Calculator_linear_regression",
          args: {
            data: [
              [1, 2],
              [2, 3.5],
            ],
          },
        },
        result: {
          content: [
            {
              type: "text",
              text: '{"slope":1.5,"intercept":0.5}',
            },
          ],
          status: "success",
        },
        state: "completed",
      },
    ];

    expect(toolCallsForMessage(["call-1"], toolCalls)).toMatchObject([
      {
        callId: "call-1",
        input: {
          data: [
            [1, 2],
            [2, 3.5],
          ],
        },
        name: "Calculator_linear_regression",
        output: [
          {
            type: "text",
            text: '{"slope":1.5,"intercept":0.5}',
          },
        ],
        status: "finished",
      },
    ]);
  });
});

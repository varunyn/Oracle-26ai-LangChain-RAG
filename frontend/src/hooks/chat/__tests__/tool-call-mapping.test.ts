import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import {
  filterToolCallsForChatStatus,
  hydrateToolCallsFromMessages,
  mergeHydratedAndLiveToolCalls,
  type NativeToolCall,
  toolCallStateForStatus,
  toolCallsForMessage,
  toRenderableToolCall,
} from "../tool-call-mapping";

function toolCall(
  callId: string,
  name: string,
  overrides: Partial<NativeToolCall> = {}
): NativeToolCall {
  return {
    callId,
    namespace: [],
    name,
    input: {},
    output: null,
    status: "running",
    error: undefined,
    ...overrides,
  };
}

describe("native assembled tool-call projection", () => {
  it("maps documented assembled fields directly", () => {
    const assembled = toolCall("call-1", "knowledge_lookup", {
      input: { query: "Oracle" },
      output: { documents: ["doc-1"] },
      status: "finished",
    });

    expect(toRenderableToolCall(assembled)).toEqual({
      callId: "call-1",
      error: undefined,
      input: { query: "Oracle" },
      name: "knowledge_lookup",
      output: { documents: ["doc-1"] },
      status: "finished",
    });
  });

  it.each([
    ["running", "input-available"],
    ["finished", "output-available"],
    ["error", "output-error"],
  ] as const)("maps %s lifecycle to the tool-card state", (status, state) => {
    expect(toolCallStateForStatus(status)).toBe(state);
  });

  it("keeps the native running state visible during an active turn", () => {
    const running = toolCall("running-1", "knowledge_lookup");
    expect(filterToolCallsForChatStatus([running], "running")).toEqual([
      running,
    ]);
    expect(filterToolCallsForChatStatus([running], "reconnecting")).toEqual([
      running,
    ]);
  });

  it("removes unfinished native calls after the turn settles", () => {
    const running = toolCall("running-1", "knowledge_lookup");
    const finished = toolCall("finished-1", "list_documents", {
      output: ["doc-1"],
      status: "finished",
    });
    expect(filterToolCallsForChatStatus([running, finished], "idle")).toEqual([
      finished,
    ]);
  });
});

describe("persisted root tool-call hydration", () => {
  it("reconstructs completed calls from standard AI and tool messages", () => {
    const messages = [
      new AIMessage({
        content: ".",
        tool_calls: [
          {
            id: "persisted-call-1",
            name: "solve_equation",
            args: { equation: "x^2 - 5x + 6 = 0" },
          },
        ],
      }),
      new ToolMessage({
        content: '{"solutions":[2,3]}',
        tool_call_id: "persisted-call-1",
        name: "solve_equation",
        status: "success",
      }),
    ];

    expect(hydrateToolCallsFromMessages(messages)).toEqual([
      {
        callId: "persisted-call-1",
        namespace: [],
        name: "solve_equation",
        input: { equation: "x^2 - 5x + 6 = 0" },
        output: '{"solutions":[2,3]}',
        status: "finished",
        error: undefined,
      },
    ]);
  });

  it("reconstructs failed calls without treating the error as output", () => {
    const messages = [
      new AIMessage({
        content: ".",
        tool_calls: [
          { id: "failed-call", name: "lookup", args: { id: "missing" } },
        ],
      }),
      new ToolMessage({
        content: "Document unavailable",
        tool_call_id: "failed-call",
        name: "lookup",
        status: "error",
      }),
    ];

    expect(hydrateToolCallsFromMessages(messages)).toEqual([
      expect.objectContaining({
        callId: "failed-call",
        output: null,
        status: "error",
        error: "Document unavailable",
      }),
    ]);
  });

  it("lets newer live lifecycle state replace the checkpoint seed", () => {
    const hydrated = toolCall("call-1", "lookup", {
      output: "old result",
      status: "finished",
    });
    const live = toolCall("call-1", "lookup", {
      output: "new result",
      status: "finished",
    });

    expect(
      mergeHydratedAndLiveToolCalls(
        [hydrated, toolCall("call-2", "summarize")],
        [live]
      )
    ).toEqual([live, toolCall("call-2", "summarize")]);
  });
});

describe("native tool-call message association", () => {
  it("associates cards by the documented callId", () => {
    const toolCalls = [
      toolCall("call-1", "knowledge_lookup", {
        input: { query: "Oracle" },
      }),
      toolCall("call-2", "list_documents", {
        output: ["doc-1"],
        status: "finished",
      }),
      toolCall("call-3", "fetch_document", {
        error: "Document unavailable",
        status: "error",
      }),
    ];

    expect(toolCallsForMessage(["call-1", "call-2"], toolCalls)).toMatchObject([
      { callId: "call-1", status: "running" },
      { callId: "call-2", status: "finished" },
    ]);
    expect(toolCallsForMessage(["missing"], toolCalls)).toEqual([]);
  });

  it("preserves running, completed, and failed cards for replay", () => {
    const toolCalls = [
      toolCall("call-1", "knowledge_lookup"),
      toolCall("call-2", "list_documents", {
        output: { documents: ["doc-1"] },
        status: "finished",
      }),
      toolCall("call-3", "fetch_document", {
        error: "Document unavailable",
        status: "error",
      }),
    ];

    expect(
      toolCallsForMessage(["call-1", "call-2", "call-3"], toolCalls)
    ).toEqual([
      expect.objectContaining({ callId: "call-1", status: "running" }),
      expect.objectContaining({ callId: "call-2", status: "finished" }),
      expect.objectContaining({ callId: "call-3", status: "error" }),
    ]);
  });
});

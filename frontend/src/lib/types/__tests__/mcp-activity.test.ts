import { describe, expect, it } from "vitest";

import { projectMcpToolActivities } from "../mcp-activity";

describe("projectMcpToolActivities", () => {
  it("keeps tool lifecycle updates in execution order", () => {
    expect(
      projectMcpToolActivities([
        {
          method: "custom",
          params: {
            data: {
              name: "mcp_tool_activity",
              payload: {
                tool_run_id: "call-1",
                tool_name: "lookup",
                status: "running",
                args: { query: "invoice" },
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
                tool_name: "lookup",
                status: "finished",
                output: "found",
              },
            },
          },
        },
      ])
    ).toEqual([
      {
        toolRunId: "call-1",
        toolName: "lookup",
        serverName: null,
        status: "finished",
        args: { query: "invoice" },
        output: "found",
        error: null,
      },
    ]);
  });

  it("keeps server metadata and merges lifecycle updates by tool run id", () => {
    expect(
      projectMcpToolActivities([
        {
          method: "custom",
          params: {
            data: {
              name: "mcp_tool_activity",
              payload: {
                tool_run_id: "call-1",
                tool_name: "Calculator_linear_regression",
                server_name: "calculator",
                status: "running",
                args: {
                  data: [
                    [1, 2],
                    [2, 3.5],
                  ],
                },
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
                server_name: "calculator",
                status: "finished",
                output: "ok",
              },
            },
          },
        },
      ]),
    ).toEqual([
      {
        toolRunId: "call-1",
        toolName: "Calculator_linear_regression",
        serverName: "calculator",
        status: "finished",
        args: {
          data: [
            [1, 2],
            [2, 3.5],
          ],
        },
        output: "ok",
        error: null,
      },
    ]);
  });
});

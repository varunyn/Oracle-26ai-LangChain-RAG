import { describe, expect, it } from "vitest";

import { formatProtocolRequestError } from "../langgraph-stream-provider";

describe("formatProtocolRequestError", () => {
  it("includes the request URL so stale active thread ids can be recovered", () => {
    expect(
      formatProtocolRequestError(
        404,
        "Not Found",
        "http://localhost:2024/threads/thread-123/state"
      )
    ).toBe(
      "Protocol request failed: 404 Not Found (http://localhost:2024/threads/thread-123/state)"
    );
  });
});

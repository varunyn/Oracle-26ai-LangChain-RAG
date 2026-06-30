import { describe, expect, it } from "vitest";

import { isMissingThreadError } from "../thread-errors";

describe("isMissingThreadError", () => {
  it("matches 404 missing-thread errors for the active thread", () => {
    expect(
      isMissingThreadError(
        new Error('HTTP 404: {"detail":"Thread with ID thread-123 not found"}'),
        "thread-123"
      )
    ).toBe(true);
  });

  it("matches protocol 404 errors whose request URL contains the active thread", () => {
    expect(
      isMissingThreadError(
        new Error(
          "Protocol request failed: 404 Not Found (http://localhost:2024/threads/thread-123/state)"
        ),
        "thread-123"
      )
    ).toBe(true);
  });

  it("ignores connectivity failures", () => {
    expect(
      isMissingThreadError(
        new Error(
          "Unable to connect to LangGraph server. Please ensure the server is running and accessible. Original error: Failed to fetch"
        ),
        "thread-123"
      )
    ).toBe(false);
  });

  it("ignores missing-thread errors for a different thread", () => {
    expect(
      isMissingThreadError(
        new Error('HTTP 404: {"detail":"Thread with ID thread-999 not found"}'),
        "thread-123"
      )
    ).toBe(false);
  });
});

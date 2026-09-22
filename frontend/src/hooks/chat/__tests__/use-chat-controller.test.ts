import { describe, expect, it } from "vitest";
import { resolveChatStatus } from "@/hooks/chat/useChatController";

const state = (
  overrides: Partial<Parameters<typeof resolveChatStatus>[0]> = {}
) =>
  resolveChatStatus({
    hasError: false,
    isThreadLoading: false,
    isLoading: false,
    reconnecting: false,
    ...overrides,
  });

describe("resolveChatStatus", () => {
  it("gives terminal stream errors precedence over every transient state", () => {
    expect(
      state({ hasError: true, isThreadLoading: true, reconnecting: true })
    ).toBe("failed");
  });

  it("reports hydration before an active or reconnecting run", () => {
    expect(state({ isThreadLoading: true, isLoading: true })).toBe("hydrating");
    expect(state({ isThreadLoading: true, reconnecting: true })).toBe(
      "hydrating"
    );
  });

  it("reports reconnecting before running while the transport retries", () => {
    expect(state({ isLoading: true, reconnecting: true })).toBe("reconnecting");
  });

  it("settles to idle when no hydration, run, or error remains", () => {
    expect(state()).toBe("idle");
  });
});

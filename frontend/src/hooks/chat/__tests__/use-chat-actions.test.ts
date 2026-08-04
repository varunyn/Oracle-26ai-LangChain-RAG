import { describe, expect, it, vi } from "vitest";

import { buildSubmitOptions, stopStreamWithToast } from "../useChatActions";

describe("buildSubmitOptions", () => {
  const config = {
    configurable: {
      mode: "rag",
      model_id: "test-model",
    },
  };

  it("forks a retry from the user turn's parent checkpoint", () => {
    expect(buildSubmitOptions(config, "checkpoint-before-user-turn")).toEqual({
      config,
      forkFrom: "checkpoint-before-user-turn",
    });
  });

  it("does not add a fork target for a new user message", () => {
    expect(buildSubmitOptions(config)).toEqual({ config });
  });
});

describe("stopStreamWithToast", () => {
  it("waits for Agent Server cancellation before reporting success", async () => {
    let resolveStop: (() => void) | undefined;
    const stop = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveStop = resolve;
        })
    );
    const toast = { error: vi.fn(), success: vi.fn() };

    const stopping = stopStreamWithToast(stop, toast);

    expect(toast.success).not.toHaveBeenCalled();
    resolveStop?.();
    await stopping;

    expect(toast.success).toHaveBeenCalledWith("Generation stopped");
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("reports cancellation failures without a false success toast", async () => {
    const stop = vi.fn().mockRejectedValue(new Error("network failure"));
    const toast = { error: vi.fn(), success: vi.fn() };

    await stopStreamWithToast(stop, toast);

    expect(toast.success).not.toHaveBeenCalled();
    expect(toast.error).toHaveBeenCalledWith(
      "The generation could not be stopped. Please try again."
    );
  });
});

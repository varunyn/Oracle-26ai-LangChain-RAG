import { describe, expect, it } from "vitest";

import { buildSubmitOptions } from "../useChatActions";

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

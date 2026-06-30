import { afterEach, describe, expect, test, vi } from "vitest";

import {
  buildLangGraphSubmitPayload,
  resolveLanggraphApiUrl,
} from "../stream-config";

describe("resolveLanggraphApiUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  test("uses browser-public LangGraph API base when configured", () => {
    vi.stubEnv("NEXT_PUBLIC_LANGGRAPH_API_BASE", "http://localhost:2024/");

    expect(resolveLanggraphApiUrl()).toBe("http://localhost:2024");
  });

  test("defaults browser chat streams to the local LangGraph Agent Server", () => {
    vi.stubGlobal("window", {
      location: {
        origin: "http://localhost:4000",
      },
    } as Window & typeof globalThis);
    vi.stubEnv("LANGGRAPH_BACKEND_URL", "http://localhost:2024");

    expect(resolveLanggraphApiUrl()).toBe("http://localhost:2024");
  });

  test("uses server-only LangGraph backend URL outside the browser", () => {
    vi.stubEnv("LANGGRAPH_BACKEND_URL", "http://langgraph:2024/");

    expect(resolveLanggraphApiUrl()).toBe("http://langgraph:2024");
  });
});

describe("buildLangGraphSubmitPayload", () => {
  test("splits graph input messages from LangGraph run config", () => {
    expect(
      buildLangGraphSubmitPayload("hello", {
        model: "test-model",
        thread_id: "thread-1",
        session_id: "session-1",
        collection_name: "default",
        enable_reranker: true,
        enable_tracing: true,
        mode: "rag",
      })
    ).toEqual({
      input: {
        messages: [{ type: "human", content: "hello" }],
      },
      config: {
        configurable: {
          model_id: "test-model",
          collection_name: "default",
          mode: "rag",
          enable_reranker: true,
          enable_tracing: true,
          session_id: "session-1",
        },
      },
    });
  });
});

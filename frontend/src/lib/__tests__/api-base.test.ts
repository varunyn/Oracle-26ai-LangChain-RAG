import { afterEach, describe, expect, test, vi } from "vitest";

import { getClientApiBase, toApiUrl } from "../api-base";

describe("getClientApiBase", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  test("prefers the explicit browser API base", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "http://localhost:2024/");
    vi.stubGlobal("window", {
      location: {
        origin: "http://localhost:4000",
      },
    } as Window & typeof globalThis);

    expect(getClientApiBase()).toBe("http://localhost:2024");
    expect(toApiUrl("/api/config")).toBe("http://localhost:2024/api/config");
  });

  test("uses relative API paths in the browser when no base is configured", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "");
    vi.stubGlobal("window", {
      location: {
        origin: "http://localhost:4000",
      },
    } as Window & typeof globalThis);

    expect(getClientApiBase()).toBe("");
    expect(toApiUrl("/api/config")).toBe("/api/config");
  });

  test("defaults server-side product API calls to the Agent Server", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "");

    expect(getClientApiBase()).toBe("http://localhost:2024");
  });

  test("uses the internal product API base first when configured server-side", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "http://localhost:2024/");
    vi.stubEnv("FASTAPI_BACKEND_URL", "http://langgraph:2024/");

    expect(getClientApiBase()).toBe("http://langgraph:2024");
    expect(toApiUrl("api/config")).toBe("http://langgraph:2024/api/config");
  });
});

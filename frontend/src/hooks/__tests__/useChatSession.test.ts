import { afterEach, describe, expect, it, vi } from "vitest";
import { THREAD_ID_STORAGE_KEY } from "@/constants/chat";
import {
  createInitialState,
  loadThreadHistory,
} from "../useChatSession";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubLocalStorage(values: Record<string, string | null>) {
  vi.stubGlobal("window", {
    localStorage: {
      getItem: (key: string) => values[key] ?? null,
    },
  });
}

describe("loadThreadHistory", () => {
  it("derives thread summaries from server-side thread search results", async () => {
    const history = await loadThreadHistory({
      threads: {
        search: async () => [
          {
            thread_id: "thread-2",
            created_at: "2026-06-25T10:00:00Z",
            updated_at: "2026-06-26T10:00:00Z",
            values: {
              messages: [
                { type: "human", content: "Can you tell me about net payment terms?" },
              ],
            },
          },
          {
            thread_id: "thread-1",
            created_at: "2026-06-24T10:00:00Z",
            updated_at: "2026-06-24T11:00:00Z",
            values: {},
          },
        ],
      },
    });

    expect(history).toEqual([
      {
        id: "thread-2",
        title: "Can you tell me about net payment terms?",
        createdAt: Date.parse("2026-06-25T10:00:00Z"),
        updatedAt: Date.parse("2026-06-26T10:00:00Z"),
      },
      {
        id: "thread-1",
        title: "Chat read-1",
        createdAt: Date.parse("2026-06-24T10:00:00Z"),
        updatedAt: Date.parse("2026-06-24T11:00:00Z"),
      },
    ]);
  });

});

describe("createInitialState", () => {
  it("starts new sessions unbound so LangGraph can create the server thread", () => {
    stubLocalStorage({});

    expect(createInitialState()).toEqual({
      threadId: null,
      threadHistory: [],
      hydrated: true,
    });
  });

  it("rehydrates only an existing stored server thread id", () => {
    stubLocalStorage({ [THREAD_ID_STORAGE_KEY]: " thread-from-server " });

    expect(createInitialState()).toEqual({
      threadId: "thread-from-server",
      threadHistory: [],
      hydrated: true,
    });
  });

  it("does not rehydrate sidebar history from localStorage", () => {
    stubLocalStorage({
      [THREAD_ID_STORAGE_KEY]: "thread-from-server",
      "legacy-rag-agent-chat-threads": JSON.stringify([
        { id: "deleted-thread", title: "Deleted", createdAt: 1, updatedAt: 1 },
      ]),
    });

    expect(createInitialState()).toEqual({
      threadId: "thread-from-server",
      threadHistory: [],
      hydrated: true,
    });
  });
});

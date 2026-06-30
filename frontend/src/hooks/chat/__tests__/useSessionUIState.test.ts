import { describe, expect, it } from "vitest";

import { deriveCollectionState } from "../useSessionUIState";

describe("deriveCollectionState", () => {
  it("does not fall back to a hardcoded collection when app config is missing", () => {
    expect(deriveCollectionState(null, "")).toEqual({
      collectionList: [],
      collectionName: "",
    });
  });

  it("uses the configured collection when the current selection is empty", () => {
    expect(
      deriveCollectionState({ collection_list: ["ORACLE_WEB_EMBEDDINGS"] }, "")
    ).toEqual({
      collectionList: ["ORACLE_WEB_EMBEDDINGS"],
      collectionName: "ORACLE_WEB_EMBEDDINGS",
    });
  });

  it("keeps the current collection when it is still valid", () => {
    expect(
      deriveCollectionState(
        { collection_list: ["ORACLE_WEB_EMBEDDINGS", "SECONDARY"] },
        "SECONDARY"
      )
    ).toEqual({
      collectionList: ["ORACLE_WEB_EMBEDDINGS", "SECONDARY"],
      collectionName: "SECONDARY",
    });
  });
});

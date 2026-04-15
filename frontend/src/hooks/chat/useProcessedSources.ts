import { useCallback, useEffect, useState } from "react";
import { toApiUrl } from "@/lib/api-base";

export type ProcessedSource = {
  source: string;
  chunk_count: number;
};

export function useProcessedSources(collectionName: string) {
  const [sources, setSources] = useState<ProcessedSource[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deletingSource, setDeletingSource] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (collectionName) {
        params.set("collection_name", collectionName);
      }
      const query = params.toString();
      const response = await fetch(
        `${toApiUrl("/api/documents/sources")}${query ? `?${query}` : ""}`,
      );
      const data = (await response.json()) as {
        error?: string;
        sources?: ProcessedSource[];
      };

      if (!response.ok || (typeof data.error === "string" && data.error.length > 0)) {
        setError(data.error ?? "We couldn't load processed sources.");
        setSources([]);
        return;
      }

      setSources(Array.isArray(data.sources) ? data.sources : []);
    } catch {
      setError("We couldn't load processed sources. Try again in a moment.");
      setSources([]);
    } finally {
      setIsLoading(false);
    }
  }, [collectionName]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const deleteSource = useCallback(
    async (source: string) => {
      setDeletingSource(source);
      setError(null);
      try {
        const params = new URLSearchParams({ source });
        if (collectionName) {
          params.set("collection_name", collectionName);
        }
        const response = await fetch(`${toApiUrl("/api/documents/source")}?${params.toString()}`, {
          method: "DELETE",
        });
        const data = (await response.json()) as { error?: string };
        if (!response.ok || (typeof data.error === "string" && data.error.length > 0)) {
          setError(data.error ?? "We couldn't delete that source.");
          return false;
        }
        await refresh();
        return true;
      } catch {
        setError("We couldn't delete that source. Try again in a moment.");
        return false;
      } finally {
        setDeletingSource(null);
      }
    },
    [collectionName, refresh],
  );

  return {
    sources,
    isLoading,
    error,
    deletingSource,
    refresh,
    deleteSource,
  };
}

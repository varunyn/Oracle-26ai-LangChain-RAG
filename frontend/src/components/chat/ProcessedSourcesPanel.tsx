"use client";

import { RefreshCcw, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useProcessedSources } from "@/hooks/chat/useProcessedSources";

type ProcessedSourcesPanelProps = {
  collectionName: string;
};

function formatSourceLabel(source: string) {
  const trimmed = source.trim();
  if (trimmed.startsWith("file://")) {
    const value = trimmed.replace("file://", "");
    const fileName = value.split("/").pop() || value;
    return { kind: "File", value: fileName, title: value };
  }
  return { kind: "Source", value: trimmed, title: trimmed };
}

export function ProcessedSourcesPanel({
  collectionName,
}: ProcessedSourcesPanelProps): React.ReactElement {
  const { sources, isLoading, error, deletingSource, refresh, deleteSource } =
    useProcessedSources(collectionName);
  const [pendingDeleteSource, setPendingDeleteSource] = useState<string | null>(null);

  const totalChunks = useMemo(
    () => sources.reduce((sum, item) => sum + item.chunk_count, 0),
    [sources],
  );

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden bg-muted/10">
      <div className="border-b border-border bg-card px-4 py-4 sm:px-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-foreground sm:text-xl">
              Processed sources
            </h2>
            <p className="text-sm text-muted-foreground">
              Review indexed sources for the selected collection and delete all related chunks when needed.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refresh()}
            className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted/50"
          >
            <RefreshCcw className="size-4" />
            Refresh
          </button>
        </div>
        <div className="mt-4 flex flex-wrap gap-3 text-xs text-muted-foreground">
          <span className="rounded-full border border-border bg-muted/60 px-3 py-1">
            Collection: <span className="font-semibold text-foreground">{collectionName}</span>
          </span>
          <span className="rounded-full border border-border bg-muted/60 px-3 py-1">
            Sources: <span className="font-semibold text-foreground">{sources.length}</span>
          </span>
          <span className="rounded-full border border-border bg-muted/60 px-3 py-1">
            Indexed chunks: <span className="font-semibold text-foreground">{totalChunks}</span>
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto px-4 py-4 sm:px-6 sm:py-6">
        <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
          {error ? (
            <div className="border-b border-border bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          {isLoading ? (
            <div className="px-4 py-10 text-sm text-muted-foreground">
              Loading processed sources...
            </div>
          ) : sources.length === 0 ? (
            <div className="px-4 py-10 text-sm text-muted-foreground">
              No processed sources found for this collection yet.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-border text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-foreground">Type</th>
                    <th className="px-4 py-3 text-left font-medium text-foreground">Source</th>
                    <th className="px-4 py-3 text-right font-medium text-foreground">Chunks</th>
                    <th className="px-4 py-3 text-right font-medium text-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border bg-card">
                  {sources.map((item) => {
                    const formatted = formatSourceLabel(item.source);
                    const isPendingDelete = pendingDeleteSource === item.source;
                    const isDeleting = deletingSource === item.source;

                    return (
                      <tr key={item.source} className="align-top">
                        <td className="px-4 py-4 text-muted-foreground">
                          <span className="inline-flex rounded-full border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground">
                            {formatted.kind}
                          </span>
                        </td>
                        <td className="max-w-0 px-4 py-4 text-foreground">
                          <div className="truncate" title={formatted.title}>
                            {formatted.value}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-right font-medium text-foreground">
                          {item.chunk_count.toLocaleString()}
                        </td>
                        <td className="px-4 py-4 text-right">
                          {isPendingDelete ? (
                            <div className="ml-auto flex max-w-sm flex-col items-end gap-2">
                              <p className="text-xs leading-relaxed text-muted-foreground">
                                Delete this source and all related chunks from the collection? This action cannot be undone.
                              </p>
                              <div className="flex items-center gap-2">
                                <button
                                  type="button"
                                  onClick={() => setPendingDeleteSource(null)}
                                  disabled={isDeleting}
                                  className="rounded-md border border-input bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-muted/50 disabled:opacity-60"
                                >
                                  Cancel
                                </button>
                                <button
                                  type="button"
                                  onClick={async () => {
                                    const deleted = await deleteSource(item.source);
                                    if (deleted) {
                                      setPendingDeleteSource(null);
                                    }
                                  }}
                                  disabled={isDeleting}
                                  className="inline-flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive transition-colors hover:bg-destructive/15 disabled:opacity-60"
                                >
                                  <Trash2 className="size-3.5" />
                                  {isDeleting ? "Deleting..." : "Delete source"}
                                </button>
                              </div>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setPendingDeleteSource(item.source)}
                              disabled={Boolean(deletingSource)}
                              className="inline-flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive transition-colors hover:bg-destructive/15 disabled:opacity-60"
                            >
                              <Trash2 className="size-3.5" />
                              Delete
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

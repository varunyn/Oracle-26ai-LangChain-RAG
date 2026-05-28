"use client";

import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Loader2,
  RefreshCcw,
  Trash2,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type {
  DocumentIngestionFileStatus,
  DocumentIngestionJob,
  DocumentIngestionJobStatus,
} from "@/hooks/chat/useDocumentIngestionJobs";
import { useProcessedSources } from "@/hooks/chat/useProcessedSources";

type ProcessedSourcesPanelProps = {
  collectionName: string;
  ingestionJobs: DocumentIngestionJob[];
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

const jobStatusLabel: Record<DocumentIngestionJobStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  interrupted: "Interrupted",
};

const fileStatusLabel: Record<DocumentIngestionFileStatus, string> = {
  queued: "Queued",
  parsing: "Parsing",
  embedding: "Embedding",
  indexed: "Indexed",
  failed: "Failed",
  interrupted: "Interrupted",
};

function statusClassName(status: DocumentIngestionJobStatus | DocumentIngestionFileStatus) {
  if (status === "completed" || status === "indexed") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700";
  }
  if (status === "failed" || status === "interrupted") {
    return "border-destructive/40 bg-destructive/10 text-destructive";
  }
  if (status === "embedding" || status === "parsing" || status === "running") {
    return "border-blue-500/30 bg-blue-500/10 text-blue-700";
  }
  return "border-border bg-muted/50 text-muted-foreground";
}

function StatusIcon({
  status,
}: {
  status: DocumentIngestionJobStatus | DocumentIngestionFileStatus;
}) {
  if (status === "completed" || status === "indexed") {
    return <CheckCircle2 className="size-3.5" aria-hidden />;
  }
  if (status === "failed") {
    return <XCircle className="size-3.5" aria-hidden />;
  }
  if (status === "interrupted") {
    return <AlertTriangle className="size-3.5" aria-hidden />;
  }
  if (status === "embedding" || status === "parsing" || status === "running") {
    return <Loader2 className="size-3.5 animate-spin" aria-hidden />;
  }
  return <CircleDashed className="size-3.5" aria-hidden />;
}

function formatTimestamp(value: string | undefined) {
  if (!value) return "Waiting";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Updated";
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function IngestionActivity({
  jobs,
}: {
  jobs: DocumentIngestionJob[];
}): React.ReactElement | null {
  if (jobs.length === 0) return null;

  return (
    <div className="mb-4 overflow-hidden rounded-lg border border-border bg-card">
      <div className="flex flex-col gap-1 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Ingestion activity</h3>
          <p className="text-xs text-muted-foreground">
            File-level indexing status for recent uploads in this collection.
          </p>
        </div>
        <span className="text-xs text-muted-foreground">
          {jobs.length} recent job{jobs.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="divide-y divide-border">
        {jobs.map((job) => (
          <div key={job.job_id} className="px-4 py-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${statusClassName(job.status)}`}
                  >
                    <StatusIcon status={job.status} />
                    {jobStatusLabel[job.status]}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {job.files_processed} file{job.files_processed === 1 ? "" : "s"} indexed
                    {" - "}
                    {job.chunks_added.toLocaleString()} chunks
                  </span>
                </div>
                {job.error ? (
                  <p className="mt-1 text-xs leading-5 text-destructive">{job.error}</p>
                ) : null}
              </div>
              <span className="shrink-0 text-xs text-muted-foreground">
                {formatTimestamp(job.updated_at)}
              </span>
            </div>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full text-xs">
                <tbody className="divide-y divide-border/70">
                  {job.files.map((file) => (
                    <tr key={file.file_id}>
                      <td className="max-w-0 py-2 pr-3 text-foreground">
                        <div className="truncate" title={file.name}>
                          {file.name}
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-3 py-2">
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-medium ${statusClassName(file.status)}`}
                        >
                          <StatusIcon status={file.status} />
                          {fileStatusLabel[file.status]}
                        </span>
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-right text-muted-foreground">
                        {file.chunks_added.toLocaleString()} chunks
                      </td>
                      <td className="max-w-[18rem] py-2 pl-3 text-right text-muted-foreground">
                        <span className="block truncate" title={file.message ?? undefined}>
                          {file.message ?? formatTimestamp(file.completed_at ?? file.started_at ?? undefined)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ProcessedSourcesPanel({
  collectionName,
  ingestionJobs,
}: ProcessedSourcesPanelProps): React.ReactElement {
  const { sources, isLoading, error, deletingSource, refresh, deleteSource } =
    useProcessedSources(collectionName);
  const [pendingDeleteSource, setPendingDeleteSource] = useState<string | null>(null);

  const totalChunks = useMemo(
    () => sources.reduce((sum, item) => sum + item.chunk_count, 0),
    [sources],
  );
  const collectionJobs = useMemo(
    () => ingestionJobs.filter((job) => job.collection === collectionName),
    [collectionName, ingestionJobs],
  );
  const completedJobKey = useMemo(
    () =>
      collectionJobs
        .filter((job) => job.status === "completed")
        .map((job) => `${job.job_id}:${job.updated_at ?? ""}`)
        .join("|"),
    [collectionJobs],
  );

  useEffect(() => {
    if (!completedJobKey) return;
    void refresh();
  }, [completedJobKey, refresh]);

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
        <IngestionActivity jobs={collectionJobs} />

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

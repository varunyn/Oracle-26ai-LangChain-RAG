import { useCallback, useEffect, useMemo, useState } from "react";
import { toApiUrl } from "@/lib/api-base";

export type DocumentIngestionFileStatus =
  | "queued"
  | "parsing"
  | "embedding"
  | "indexed"
  | "failed"
  | "interrupted";

export type DocumentIngestionJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export type DocumentIngestionFile = {
  file_id: string;
  name: string;
  status: DocumentIngestionFileStatus;
  chunks_added: number;
  message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
};

export type DocumentIngestionJob = {
  job_id: string;
  collection: string;
  status: DocumentIngestionJobStatus;
  chunks_added: number;
  files_processed: number;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
  files: DocumentIngestionFile[];
};

const TERMINAL_JOB_STATUSES = new Set<DocumentIngestionJobStatus>([
  "completed",
  "failed",
  "interrupted",
]);

function mergeJob(
  currentJobs: DocumentIngestionJob[],
  nextJob: DocumentIngestionJob,
): DocumentIngestionJob[] {
  const remainingJobs = currentJobs.filter((job) => job.job_id !== nextJob.job_id);
  return [nextJob, ...remainingJobs].slice(0, 12);
}

export function useDocumentIngestionJobs() {
  const [jobs, setJobs] = useState<DocumentIngestionJob[]>([]);

  const trackJob = useCallback((job: DocumentIngestionJob) => {
    setJobs((currentJobs) => mergeJob(currentJobs, job));
  }, []);

  const activeJobIdsKey = useMemo(
    () =>
      jobs
        .filter((job) => !TERMINAL_JOB_STATUSES.has(job.status))
        .map((job) => job.job_id)
        .join("|"),
    [jobs],
  );

  useEffect(() => {
    if (!activeJobIdsKey) return;

    let cancelled = false;
    const activeJobIds = activeJobIdsKey.split("|");
    const refreshActiveJobs = async () => {
      const updates = await Promise.all(
        activeJobIds.map(async (jobId) => {
          try {
            const response = await fetch(
              toApiUrl(`/api/documents/jobs/${encodeURIComponent(jobId)}`),
            );
            if (!response.ok) return null;
            return (await response.json()) as DocumentIngestionJob;
          } catch {
            return null;
          }
        }),
      );

      if (cancelled) return;
      const validUpdates = updates.filter(
        (job): job is DocumentIngestionJob => job !== null,
      );
      if (validUpdates.length === 0) return;

      setJobs((currentJobs) =>
        validUpdates.reduce(
          (mergedJobs, job) => mergeJob(mergedJobs, job),
          currentJobs,
        ),
      );
    };

    void refreshActiveJobs();
    const intervalId = window.setInterval(() => {
      void refreshActiveJobs();
    }, 1800);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeJobIdsKey]);

  return {
    jobs,
    trackJob,
  };
}

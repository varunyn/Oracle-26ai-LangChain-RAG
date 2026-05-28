import { useCallback, useState } from "react";
import {
  type DocumentIngestionJob,
  useDocumentIngestionJobs,
} from "@/hooks/chat/useDocumentIngestionJobs";
import { toApiUrl } from "@/lib/api-base";

export function useChatMutations(collectionName: string) {
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { jobs, trackJob } = useDocumentIngestionJobs();

  const handleUpload = useCallback(async () => {
    if (!uploadFiles.length || isUploading) return;
    setIsUploading(true);
    setUploadStatus("Uploading documents to the selected collection...");

    const formData = new FormData();
    uploadFiles.forEach((file) => {
      formData.append("files", file);
    });
    if (collectionName) formData.append("collection_name", collectionName);

    try {
      const res = await fetch(toApiUrl("/api/documents/upload"), {
        method: "POST",
        body: formData,
      });
      const data = (await res.json()) as {
        error?: string;
        job_id?: string;
        status?: DocumentIngestionJob["status"];
        files?: DocumentIngestionJob["files"];
        chunks_added?: number;
        collection?: string;
        files_processed?: number;
      };

      if (typeof data.error === "string" && data.error.length > 0) {
        setUploadStatus(`We couldn't add your documents: ${data.error}`);
        return;
      }

      if (typeof data.job_id === "string" && data.job_id.length > 0) {
        trackJob({
          job_id: data.job_id,
          collection: data.collection ?? collectionName,
          status: data.status ?? "queued",
          chunks_added: data.chunks_added ?? 0,
          files_processed: data.files_processed ?? 0,
          files: Array.isArray(data.files) ? data.files : [],
        });
        setUploadStatus(
          `Started indexing ${uploadFiles.length} file${uploadFiles.length === 1 ? "" : "s"}. Open Processed sources to watch progress.`,
        );
        setUploadFiles([]);
        return;
      }

      setUploadStatus(
        `Added ${data.files_processed ?? uploadFiles.length} file(s) to ${data.collection ?? collectionName ?? "the selected collection"} and indexed ${data.chunks_added ?? 0} chunks for retrieval.`,
      );
      setUploadFiles([]);
    } catch {
      setUploadStatus(
        "We couldn't upload your documents. Try again in a moment.",
      );
    } finally {
      setIsUploading(false);
    }
  }, [collectionName, isUploading, trackJob, uploadFiles]);

  return {
    uploadFiles,
    setUploadFiles,
    uploadStatus,
    isUploading,
    handleUpload,
    ingestionJobs: jobs,
  };
}

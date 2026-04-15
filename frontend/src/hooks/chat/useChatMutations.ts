import { useCallback, useState } from "react";
import { toApiUrl } from "@/lib/api-base";

export function useChatMutations(collectionName: string) {
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const handleUpload = useCallback(async () => {
    if (!uploadFiles.length) return;
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
        chunks_added?: number;
        collection?: string;
        files_processed?: number;
      };

      if (typeof data.error === "string" && data.error.length > 0) {
        setUploadStatus(`We couldn't add your documents: ${data.error}`);
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
    }
  }, [collectionName, uploadFiles]);

  return {
    uploadFiles,
    setUploadFiles,
    uploadStatus,
    handleUpload,
  };
}

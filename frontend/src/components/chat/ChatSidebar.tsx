"use client";

import { useRef, useState } from "react";
import { Upload } from "lucide-react";

type FlowMode = "rag" | "mcp" | "mixed" | "direct";

type AppConfig = {
  region?: string;
  embed_model_id?: string;
  collection_list?: string[];
} | null;

type ChatSidebarProps = {
  open: boolean;
  appConfig: AppConfig;
  collectionList: string[];
  collectionName: string;
  setCollectionName: (v: string) => void;
  flowMode: FlowMode;
  setFlowMode: (v: FlowMode) => void;
  enableReranker: boolean;
  setEnableReranker: (v: boolean) => void;
  enableTracing: boolean;
  setEnableTracing: (v: boolean) => void;
  onClearChat: () => void;
  uploadFiles: File[];
  setUploadFiles: React.Dispatch<React.SetStateAction<File[]>>;
  uploadStatus: string | null;
  onUpload: () => void;
};

export function ChatSidebar({
  open: sidebarOpen,
  appConfig,
  collectionList,
  collectionName,
  setCollectionName,
  flowMode,
  setFlowMode,
  enableReranker,
  setEnableReranker,
  enableTracing,
  setEnableTracing,
  onClearChat,
  uploadFiles,
  setUploadFiles,
  uploadStatus,
  onUpload,
}: ChatSidebarProps): React.ReactElement {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadDragActive, setUploadDragActive] = useState(false);
  const regionFieldId = "chat-sidebar-region";
  const collectionFieldId = "chat-sidebar-collection";
  const embedModelFieldId = "chat-sidebar-embed-model";
  const flowModeFieldId = "chat-sidebar-flow-mode";

  const allowedExtensions = [
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".markdown",
  ];

  const addAcceptedFiles = (files: FileList | File[]) => {
    const acceptedFiles = Array.from(files).filter((file) => {
      const extension = "." + (file.name.split(".").pop() ?? "").toLowerCase();
      return allowedExtensions.includes(extension);
    });

    if (acceptedFiles.length > 0) {
      setUploadFiles((previousFiles) => [...previousFiles, ...acceptedFiles]);
    }
  };

  const removeSelectedFile = (fileToRemove: File) => {
    setUploadFiles((previousFiles) =>
      previousFiles.filter(
        (file) =>
          !(
            file.name === fileToRemove.name &&
            file.size === fileToRemove.size &&
            file.lastModified === fileToRemove.lastModified
          ),
      ),
    );
  };

  return (
    <aside
      className={`h-full min-h-0 bg-card border-r border-border flex flex-col shrink-0 shadow-sm transition-[width] duration-200 ${
        sidebarOpen ? "w-72" : "w-0 overflow-hidden border-0"
      }`}
      aria-hidden={!sidebarOpen}
    >
      <div className="min-w-[18rem] border-b border-border px-4 py-4 sm:px-5">
        <h2 className="font-semibold text-foreground text-sm uppercase tracking-wider text-muted-foreground">
          Retrieval settings
        </h2>
      </div>
      <div className="min-w-[18rem] space-y-5 px-4 py-5 sm:px-5">
        <section className="space-y-4" aria-label="RAG settings">
          <div>
            <label
              htmlFor={regionFieldId}
              className="mb-1.5 block text-xs font-medium text-muted-foreground"
            >
              Region
            </label>
            <input
              id={regionFieldId}
              type="text"
              value={appConfig?.region ?? "—"}
              readOnly
              aria-readonly
              className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-sm text-foreground"
            />
          </div>
          <div>
            <label
              htmlFor={collectionFieldId}
              className="mb-1.5 block text-xs font-medium text-muted-foreground"
            >
              Collection
            </label>
            <select
              id={collectionFieldId}
              value={collectionName}
              onChange={(e) => setCollectionName(e.target.value)}
              data-testid="chat-collection-select"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {collectionList.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label
              htmlFor={embedModelFieldId}
              className="mb-1.5 block text-xs font-medium text-muted-foreground"
            >
              Embed model
            </label>
            <input
              id={embedModelFieldId}
              type="text"
              value={appConfig?.embed_model_id ?? "—"}
              readOnly
              aria-readonly
              className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-sm text-foreground"
            />
          </div>
          <div className="flex flex-col gap-3 border-t border-border/60 pt-4">
            <div>
              <label
                htmlFor={flowModeFieldId}
                className="mb-1.5 block text-sm text-foreground"
              >
                Flow mode
              </label>
              <select
                id={flowModeFieldId}
                value={flowMode}
                onChange={(e) => setFlowMode(e.target.value as FlowMode)}
                className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-sm text-foreground"
              >
                <option value="rag">RAG only</option>
                <option value="mcp">MCP tools only</option>
                <option value="mixed">Mixed (RAG + MCP)</option>
                <option value="direct">Direct (no RAG, no tools)</option>
              </select>
            </div>
            <label className="flex items-center gap-3 rounded-md py-0.5">
              <input
                type="checkbox"
                checked={enableReranker}
                onChange={(e) => setEnableReranker(e.target.checked)}
                className="rounded border-input text-primary focus:ring-ring"
              />
              <span className="text-sm text-foreground">Enable Reranker</span>
            </label>
            <label className="flex items-center gap-3 rounded-md py-0.5">
              <input
                type="checkbox"
                checked={enableTracing}
                onChange={(e) => setEnableTracing(e.target.checked)}
                className="rounded border-input text-primary focus:ring-ring"
              />
              <span className="text-sm text-foreground">Enable tracing</span>
            </label>
          </div>
        </section>
        <button
          type="button"
          onClick={onClearChat}
          data-testid="chat-clear-history"
          className="w-full rounded-md border border-border bg-secondary px-3 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-secondary/80"
        >
          Clear Chat History
        </button>
      </div>
      <div className="mt-auto min-w-[18rem] border-t border-border px-4 py-5 sm:px-5">
        <h3 className="mb-3 text-sm font-medium text-foreground">
          Upload documents
        </h3>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setUploadDragActive(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setUploadDragActive(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setUploadDragActive(false);
            addAcceptedFiles(e.dataTransfer.files);
          }}
          className={`flex min-h-32 w-full flex-col items-center justify-center gap-2.5 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${
            uploadDragActive
              ? "border-primary bg-primary/10 text-primary"
              : "border-border bg-muted/30 text-muted-foreground hover:border-muted-foreground/50 hover:bg-muted/50"
          }`}
          aria-label="Add documents to the current collection"
        >
          <Upload className="size-8 shrink-0" aria-hidden />
          <span className="text-xs font-medium">
            Drag documents here, or select files to add to this collection
          </span>
          <span className="text-xs opacity-80">
            Supported formats: PDF, HTML, TXT, and Markdown
          </span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.html,.htm,.txt,.md"
          onChange={(e) => {
            if (e.target.files) {
              addAcceptedFiles(e.target.files);
              e.target.value = "";
            }
          }}
          className="sr-only"
          aria-hidden
        />
        {uploadFiles.length > 0 ? (
          <div className="mt-3 rounded-md border border-border/60 bg-muted/20 p-3">
            <p className="text-xs font-medium text-foreground">
              Selected {uploadFiles.length} file{uploadFiles.length === 1 ? "" : "s"}
            </p>
            <ul className="mt-2 max-h-28 space-y-1 overflow-y-auto pr-1 text-xs text-muted-foreground">
              {uploadFiles.map((file) => (
                <li
                  key={`${file.name}-${file.size}-${file.lastModified}`}
                  className="flex items-start gap-2"
                >
                  <span className="min-w-0 flex-1 truncate" title={file.name}>
                    {file.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeSelectedFile(file)}
                    className="shrink-0 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
                    aria-label={`Remove ${file.name}`}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {uploadFiles.length > 0 ? (
          <button
            type="button"
            onClick={onUpload}
            className="mt-3 min-h-11 w-full rounded-md bg-primary px-3 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Add documents to collection
          </button>
        ) : null}
        {uploadStatus ? (
          <p className="mt-3 text-xs leading-5 text-muted-foreground" role="status">
            {uploadStatus}
          </p>
        ) : null}
      </div>
    </aside>
  );
}

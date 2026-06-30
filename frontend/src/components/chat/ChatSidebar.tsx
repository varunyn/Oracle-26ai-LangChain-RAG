"use client";

import { MessageSquare, Plus, Upload } from "lucide-react";
import { useRef, useState } from "react";
import type { ChatThreadSummary } from "@/hooks/useChatSession";

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
  isUploading: boolean;
  onUpload: () => void;
  threadHistory: ChatThreadSummary[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onNewChat: () => void;
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
  isUploading,
  onUpload,
  threadHistory,
  activeThreadId,
  onSelectThread,
  onNewChat,
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
          )
      )
    );
  };

  return (
    <aside
      aria-hidden={!sidebarOpen}
      className={`flex h-full min-h-0 shrink-0 flex-col border-border border-r bg-card shadow-sm transition-[width] duration-200 ${
        sidebarOpen ? "w-72" : "w-0 overflow-hidden border-0"
      }`}
    >
      <div className="min-w-[18rem] border-border border-b px-4 py-4 sm:px-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="font-semibold text-muted-foreground text-sm uppercase tracking-wider">
            Conversations
          </h2>
          <button
            aria-label="Start new chat"
            className="inline-flex size-8 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            onClick={onNewChat}
            title="Start new chat"
            type="button"
          >
            <Plus aria-hidden className="size-4" />
          </button>
        </div>
      </div>
      <div className="min-h-0 min-w-[18rem] flex-1 overflow-y-auto">
        <div className="border-border/70 border-b px-3 py-3">
          <div
            aria-label="Chat history"
            className="max-h-[min(45vh,28rem)] space-y-1 overflow-y-auto pr-1"
            data-testid="chat-history-list"
          >
            {threadHistory.map((thread) => {
              const active = thread.id === activeThreadId;
              return (
                <button
                  aria-current={active ? "page" : undefined}
                  className={`flex w-full min-w-0 items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors ${
                    active
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                  }`}
                  data-testid="chat-history-thread"
                  key={thread.id}
                  onClick={() => onSelectThread(thread.id)}
                  type="button"
                >
                  <MessageSquare aria-hidden className="size-4 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">
                    {thread.title}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="space-y-5 px-4 py-5 sm:px-5">
          <h2 className="font-semibold text-muted-foreground text-sm uppercase tracking-wider">
            Retrieval settings
          </h2>
          <section aria-label="RAG settings" className="space-y-4">
            <div>
              <label
                className="mb-1.5 block font-medium text-muted-foreground text-xs"
                htmlFor={regionFieldId}
              >
                Region
              </label>
              <input
                aria-readonly
                className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-foreground text-sm"
                id={regionFieldId}
                readOnly
                type="text"
                value={appConfig?.region ?? "—"}
              />
            </div>
            <div>
              <label
                className="mb-1.5 block font-medium text-muted-foreground text-xs"
                htmlFor={collectionFieldId}
              >
                Collection
              </label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                data-testid="chat-collection-select"
                disabled={collectionList.length === 0}
                id={collectionFieldId}
                onChange={(e) => setCollectionName(e.target.value)}
                value={collectionName}
              >
                {collectionList.length === 0 ? (
                  <option value="">Unavailable</option>
                ) : null}
                {collectionList.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label
                className="mb-1.5 block font-medium text-muted-foreground text-xs"
                htmlFor={embedModelFieldId}
              >
                Embed model
              </label>
              <input
                aria-readonly
                className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-foreground text-sm"
                id={embedModelFieldId}
                readOnly
                type="text"
                value={appConfig?.embed_model_id ?? "—"}
              />
            </div>
            <div className="flex flex-col gap-3 border-border/60 border-t pt-4">
              <div>
                <label
                  className="mb-1.5 block text-foreground text-sm"
                  htmlFor={flowModeFieldId}
                >
                  Flow mode
                </label>
                <select
                  className="w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-foreground text-sm"
                  id={flowModeFieldId}
                  onChange={(e) => setFlowMode(e.target.value as FlowMode)}
                  value={flowMode}
                >
                  <option value="rag">RAG only</option>
                  <option value="mcp">MCP tools only</option>
                  <option value="mixed">Mixed (RAG + MCP)</option>
                  <option value="direct">Direct (no RAG, no tools)</option>
                </select>
              </div>
              <label className="flex items-center gap-3 rounded-md py-0.5">
                <input
                  checked={enableReranker}
                  className="rounded border-input text-primary focus:ring-ring"
                  onChange={(e) => setEnableReranker(e.target.checked)}
                  type="checkbox"
                />
                <span className="text-foreground text-sm">Enable Reranker</span>
              </label>
              <label className="flex items-center gap-3 rounded-md py-0.5">
                <input
                  checked={enableTracing}
                  className="rounded border-input text-primary focus:ring-ring"
                  onChange={(e) => setEnableTracing(e.target.checked)}
                  type="checkbox"
                />
                <span className="text-foreground text-sm">Enable tracing</span>
              </label>
            </div>
          </section>
          <button
            className="w-full rounded-md border border-border bg-secondary px-3 py-2.5 font-medium text-foreground text-sm transition-colors hover:bg-secondary/80"
            data-testid="chat-clear-history"
            onClick={onClearChat}
            type="button"
          >
            Clear Chat History
          </button>
        </div>
        <div className="border-border border-t px-4 py-5 sm:px-5">
          <h3 className="mb-3 font-medium text-foreground text-sm">
            Upload documents
          </h3>
          <button
            aria-label="Add documents to the current collection"
            className={`flex min-h-32 w-full flex-col items-center justify-center gap-2.5 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${
              uploadDragActive
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-muted/30 text-muted-foreground hover:border-muted-foreground/50 hover:bg-muted/50"
            }`}
            onClick={() => fileInputRef.current?.click()}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setUploadDragActive(false);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setUploadDragActive(true);
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setUploadDragActive(false);
              addAcceptedFiles(e.dataTransfer.files);
            }}
            type="button"
          >
            <Upload aria-hidden className="size-8 shrink-0" />
            <span className="font-medium text-xs">
              Drag documents here, or select files to add to this collection
            </span>
            <span className="text-xs opacity-80">
              Supported formats: PDF, HTML, TXT, and Markdown
            </span>
          </button>
          <input
            accept=".pdf,.html,.htm,.txt,.md"
            aria-hidden
            className="sr-only"
            multiple
            onChange={(e) => {
              if (e.target.files) {
                addAcceptedFiles(e.target.files);
                e.target.value = "";
              }
            }}
            ref={fileInputRef}
            type="file"
          />
          {uploadFiles.length > 0 ? (
            <div className="mt-3 rounded-md border border-border/60 bg-muted/20 p-3">
              <p className="font-medium text-foreground text-xs">
                Selected {uploadFiles.length} file
                {uploadFiles.length === 1 ? "" : "s"}
              </p>
              <ul className="mt-2 max-h-28 space-y-1 overflow-y-auto pr-1 text-muted-foreground text-xs">
                {uploadFiles.map((file) => (
                  <li
                    className="flex items-start gap-2"
                    key={`${file.name}-${file.size}-${file.lastModified}`}
                  >
                    <span className="min-w-0 flex-1 truncate" title={file.name}>
                      {file.name}
                    </span>
                    <button
                      aria-label={`Remove ${file.name}`}
                      className="shrink-0 font-medium text-[11px] text-muted-foreground transition-colors hover:text-foreground"
                      onClick={() => removeSelectedFile(file)}
                      type="button"
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
              className="mt-3 min-h-11 w-full rounded-md bg-primary px-3 py-2.5 font-medium text-primary-foreground text-sm transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-70"
              disabled={isUploading}
              onClick={onUpload}
              type="button"
            >
              {isUploading
                ? "Starting indexing..."
                : "Add documents to collection"}
            </button>
          ) : null}
          {uploadStatus ? (
            <p
              className="mt-3 text-muted-foreground text-xs leading-5"
              role="status"
            >
              {uploadStatus}
            </p>
          ) : null}
        </div>
      </div>
    </aside>
  );
}

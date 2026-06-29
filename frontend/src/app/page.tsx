"use client";

import dynamic from "next/dynamic";
import { Database, MessagesSquare } from "lucide-react";
import { useEffect, useState } from "react";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatInputBar } from "@/components/chat/ChatInputBar";
import { useChatSession } from "@/hooks/useChatSession";
import { useChatController } from "@/hooks/chat/useChatController";
import { useChatMutations } from "@/hooks/chat/useChatMutations";
import { useSessionUIState } from "@/hooks/chat/useSessionUIState";
import { useAppConfig } from "@/components/config-provider";
import { useToast } from "@/components/toaster";
import { ProcessedSourcesPanel } from "@/components/chat/ProcessedSourcesPanel";
import { LangGraphStreamProvider } from "@/providers/langgraph-stream-provider";
import { useLangGraphStream } from "@/providers/langgraph-stream-provider";

const ChatSidebar = dynamic(
  () => import("@/components/chat/ChatSidebar").then((mod) => mod.ChatSidebar),
  { ssr: false },
);

const ChatMessageList = dynamic(
  () =>
    import("@/components/chat/ChatMessageList").then(
      (mod) => mod.ChatMessageList,
    ),
  { ssr: false },
);

type MainView = "chat" | "sources";

export default function Chat() {
  const { config: appConfig } = useAppConfig();
  const chatSession = useChatSession();
  const {
    threadId,
    setThreadId,
    sessionId,
    clearChat,
    threadHistory,
    startNewChat,
    updateThreadTitle,
    refreshThreadHistory,
    isReady,
  } = chatSession;
  if (!isReady) {
    return <div className="h-screen bg-muted/20" aria-hidden />;
  }
  return (
    <LangGraphStreamProvider threadId={threadId} setThreadId={setThreadId}>
      <ChatPageContent
        appConfig={appConfig}
        threadId={threadId}
        sessionId={sessionId}
        clearChat={clearChat}
        threadHistory={threadHistory}
        onSelectThread={setThreadId}
        onNewChat={startNewChat}
        onUpdateThreadTitle={updateThreadTitle}
        onRefreshThreadHistory={refreshThreadHistory}
      />
    </LangGraphStreamProvider>
  );
}

type ChatPageContentProps = {
  appConfig: ReturnType<typeof useAppConfig>["config"];
  threadId: string | null;
  sessionId: string;
  clearChat: ReturnType<typeof useChatSession>["clearChat"];
  threadHistory: ReturnType<typeof useChatSession>["threadHistory"];
  onSelectThread: ReturnType<typeof useChatSession>["setThreadId"];
  onNewChat: ReturnType<typeof useChatSession>["startNewChat"];
  onUpdateThreadTitle: ReturnType<typeof useChatSession>["updateThreadTitle"];
  onRefreshThreadHistory: ReturnType<typeof useChatSession>["refreshThreadHistory"];
};

type ChatMessageLike = {
  role?: string;
  content?: string;
};

function deriveThreadTitle(messages: ChatMessageLike[]): string | null {
  const firstUserMessage = messages.find((message) => message.role === "user");
  const content = firstUserMessage?.content || "";
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) return null;
  return normalized.length > 56 ? `${normalized.slice(0, 53)}...` : normalized;
}

function ChatPageContent({
  appConfig,
  threadId,
  sessionId,
  clearChat,
  threadHistory,
  onSelectThread,
  onNewChat,
  onUpdateThreadTitle,
  onRefreshThreadHistory,
}: ChatPageContentProps) {
  const { toast } = useToast();
  const { stream } = useLangGraphStream();
  const sessionUI = useSessionUIState(appConfig);
  const [mainView, setMainView] = useState<MainView>("chat");
  const chat = useChatController({
    selectedModel: sessionUI.effectiveSelectedModel,
    threadId,
    sessionId,
    collectionName: sessionUI.collectionName,
    enableReranker: sessionUI.enableReranker,
    enableTracing: sessionUI.enableTracing,
    flowMode: sessionUI.flowMode,
    toast,
    clearSessionChat: clearChat,
  });
  const mutations = useChatMutations(sessionUI.collectionName);

  useEffect(() => {
    const title = deriveThreadTitle(chat.messages as ChatMessageLike[]);
    if (threadId && title) onUpdateThreadTitle(threadId, title);
  }, [chat.messages, onUpdateThreadTitle, threadId]);

  useEffect(() => {
    if (chat.status !== "ready") return;
    void onRefreshThreadHistory(stream.client).catch(() => undefined);
  }, [chat.status, onRefreshThreadHistory, stream.client, threadId]);

  return (
    <div
      className="flex h-screen overflow-hidden bg-muted/20"
      data-testid="chat-root"
      data-thread-id={threadId}
      data-main-view={mainView}
      data-chat-status={chat.status}
    >
      <ChatSidebar
        open={sessionUI.sidebarOpen}
        appConfig={appConfig}
        collectionList={sessionUI.collectionList}
        collectionName={sessionUI.collectionName}
        setCollectionName={sessionUI.setCollectionName}
        flowMode={sessionUI.flowMode}
        setFlowMode={sessionUI.setFlowMode}
        enableReranker={sessionUI.enableReranker}
        setEnableReranker={sessionUI.setEnableReranker}
        enableTracing={sessionUI.enableTracing}
        setEnableTracing={sessionUI.setEnableTracing}
        onClearChat={chat.handleClearChat}
        uploadFiles={mutations.uploadFiles}
        setUploadFiles={mutations.setUploadFiles}
        uploadStatus={mutations.uploadStatus}
        isUploading={mutations.isUploading}
        onUpload={mutations.handleUpload}
        threadHistory={threadHistory}
        activeThreadId={threadId}
        onSelectThread={onSelectThread}
        onNewChat={onNewChat}
      />
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col">
        <ChatHeader
          sidebarOpen={sessionUI.sidebarOpen}
          onToggleSidebar={sessionUI.toggleSidebar}
          modelList={sessionUI.modelList}
          selectedModel={sessionUI.effectiveSelectedModel}
          onSelectModel={sessionUI.handleSelectModel}
          modelSelectorOpen={sessionUI.modelSelectorOpen}
          onModelSelectorOpenChange={sessionUI.setModelSelectorOpen}
          contextUsage={chat.contextUsage}
          selectedModelData={sessionUI.selectedModelData}
        />
        <div className="border-b border-border bg-card/70 px-4 py-3 sm:px-6">
          <div className="inline-flex rounded-lg border border-border bg-background p-1 shadow-sm">
            <button
              type="button"
              onClick={() => setMainView("chat")}
              className={`inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                mainView === "chat"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              }`}
            >
              <MessagesSquare className="size-4" />
              Chat
            </button>
            <button
              type="button"
              onClick={() => setMainView("sources")}
              className={`inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                mainView === "sources"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              }`}
            >
              <Database className="size-4" />
              Processed sources
            </button>
          </div>
        </div>
        {mainView === "chat" ? (
          <>
            <ChatMessageList
              messages={chat.messages}
              toolCalls={chat.toolCalls}
              status={chat.status}
              maxCitationsToShow={chat.maxCitationsToShow}
              chatContainerRef={chat.chatContainerRef}
              onRetry={chat.handleRetry}
              onRecoverDirect={chat.handleRecoverDirect}
              onRecoverRagOnly={chat.handleRecoverRagOnly}
              onFeedback={chat.handleFeedback}
              feedbackSubmittedMessageIndexes={chat.feedbackSubmittedMessageIndexes}
              enableUserFeedback={appConfig?.enable_user_feedback}
            />
            <ChatInputBar
              input={chat.input}
              setInput={chat.setInput}
              onSubmit={chat.handleSubmit}
              status={chat.status}
              canStopStream={chat.canStopStream}
              canResumeTurn={chat.canResumeTurn}
              onStopStream={chat.handleStopStream}
              onResumeTurn={chat.handleResumeTurn}
              dynamicSuggestions={chat.dynamicSuggestions}
              suggestionsLoading={chat.suggestionsLoading}
              pendingSuggestion={chat.pendingSuggestion}
              onSuggestionClick={chat.handleSuggestionClick}
            />
          </>
        ) : (
          <ProcessedSourcesPanel
            collectionName={sessionUI.collectionName}
            ingestionJobs={mutations.ingestionJobs}
          />
        )}
      </div>
    </div>
  );
}

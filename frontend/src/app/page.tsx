"use client";

import { Database, MessagesSquare } from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatInputBar } from "@/components/chat/ChatInputBar";
import { ProcessedSourcesPanel } from "@/components/chat/ProcessedSourcesPanel";
import { useAppConfig } from "@/components/config-provider";
import { useToast } from "@/components/toaster";
import { useChatController } from "@/hooks/chat/useChatController";
import { useChatMutations } from "@/hooks/chat/useChatMutations";
import { useSessionUIState } from "@/hooks/chat/useSessionUIState";
import { useChatSession } from "@/hooks/useChatSession";
import {
  LangGraphStreamProvider,
  useLangGraphStream,
} from "@/providers/langgraph-stream-provider";

const ChatSidebar = dynamic(
  () => import("@/components/chat/ChatSidebar").then((mod) => mod.ChatSidebar),
  { ssr: false }
);

const ChatMessageList = dynamic(
  () =>
    import("@/components/chat/ChatMessageList").then(
      (mod) => mod.ChatMessageList
    ),
  { ssr: false }
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
    return <div aria-hidden className="h-screen bg-muted/20" />;
  }
  return (
    <LangGraphStreamProvider setThreadId={setThreadId} threadId={threadId}>
      <ChatPageContent
        appConfig={appConfig}
        clearChat={clearChat}
        onNewChat={startNewChat}
        onRefreshThreadHistory={refreshThreadHistory}
        onSelectThread={setThreadId}
        onUpdateThreadTitle={updateThreadTitle}
        sessionId={sessionId}
        threadHistory={threadHistory}
        threadId={threadId}
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
  onRefreshThreadHistory: ReturnType<
    typeof useChatSession
  >["refreshThreadHistory"];
};

type ChatMessageLike = {
  role?: string;
  content?: string;
};

function deriveThreadTitle(messages: ChatMessageLike[]): string | null {
  const firstUserMessage = messages.find((message) => message.role === "user");
  const content = firstUserMessage?.content || "";
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return null;
  }
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
    if (threadId && title) {
      onUpdateThreadTitle(threadId, title);
    }
  }, [chat.messages, onUpdateThreadTitle, threadId]);

  useEffect(() => {
    if (chat.status !== "ready") {
      return;
    }
    void onRefreshThreadHistory(stream.client).catch(() => undefined);
  }, [chat.status, onRefreshThreadHistory, stream.client, threadId]);

  return (
    <div
      className="flex h-screen overflow-hidden bg-muted/20"
      data-chat-status={chat.status}
      data-main-view={mainView}
      data-testid="chat-root"
      data-thread-id={threadId}
    >
      <ChatSidebar
        activeThreadId={threadId}
        appConfig={appConfig}
        collectionList={sessionUI.collectionList}
        collectionName={sessionUI.collectionName}
        enableReranker={sessionUI.enableReranker}
        enableTracing={sessionUI.enableTracing}
        flowMode={sessionUI.flowMode}
        isUploading={mutations.isUploading}
        onClearChat={chat.handleClearChat}
        onNewChat={onNewChat}
        onSelectThread={onSelectThread}
        onUpload={mutations.handleUpload}
        open={sessionUI.sidebarOpen}
        setCollectionName={sessionUI.setCollectionName}
        setEnableReranker={sessionUI.setEnableReranker}
        setEnableTracing={sessionUI.setEnableTracing}
        setFlowMode={sessionUI.setFlowMode}
        setUploadFiles={mutations.setUploadFiles}
        threadHistory={threadHistory}
        uploadFiles={mutations.uploadFiles}
        uploadStatus={mutations.uploadStatus}
      />
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden">
        <ChatHeader
          contextUsage={chat.contextUsage}
          modelList={sessionUI.modelList}
          modelSelectorOpen={sessionUI.modelSelectorOpen}
          onModelSelectorOpenChange={sessionUI.setModelSelectorOpen}
          onSelectModel={sessionUI.handleSelectModel}
          onToggleSidebar={sessionUI.toggleSidebar}
          selectedModel={sessionUI.effectiveSelectedModel}
          selectedModelData={sessionUI.selectedModelData}
          sidebarOpen={sessionUI.sidebarOpen}
        />
        <div className="border-border border-b bg-card/70 px-4 py-3 sm:px-6">
          <div className="inline-flex rounded-lg border border-border bg-background p-1 shadow-sm">
            <button
              className={`inline-flex items-center gap-2 rounded-md px-3 py-2 font-medium text-sm transition-colors ${
                mainView === "chat"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              }`}
              onClick={() => setMainView("chat")}
              type="button"
            >
              <MessagesSquare className="size-4" />
              Chat
            </button>
            <button
              className={`inline-flex items-center gap-2 rounded-md px-3 py-2 font-medium text-sm transition-colors ${
                mainView === "sources"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              }`}
              onClick={() => setMainView("sources")}
              type="button"
            >
              <Database className="size-4" />
              Processed sources
            </button>
          </div>
        </div>
        {mainView === "chat" ? (
          <>
            <ChatMessageList
              enableUserFeedback={appConfig?.enable_user_feedback}
              feedbackSubmittedMessageIndexes={
                chat.feedbackSubmittedMessageIndexes
              }
              maxCitationsToShow={chat.maxCitationsToShow}
              messages={chat.messages}
              onFeedback={chat.handleFeedback}
              onRecoverDirect={chat.handleRecoverDirect}
              onRecoverRagOnly={chat.handleRecoverRagOnly}
              onRetry={chat.handleRetry}
              progress={chat.progress}
              status={chat.status}
              toolCalls={chat.toolCalls}
            />
            <ChatInputBar
              canResumeTurn={chat.canResumeTurn}
              canStopStream={chat.canStopStream}
              dynamicSuggestions={chat.dynamicSuggestions}
              input={chat.input}
              onResumeTurn={chat.handleResumeTurn}
              onStopStream={chat.handleStopStream}
              onSubmit={chat.handleSubmit}
              onSuggestionClick={chat.handleSuggestionClick}
              pendingSuggestion={chat.pendingSuggestion}
              setInput={chat.setInput}
              status={chat.status}
              suggestionsLoading={chat.suggestionsLoading}
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

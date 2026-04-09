"use client";

import dynamic from "next/dynamic";
import { Database, MessagesSquare } from "lucide-react";
import { useState } from "react";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatInputBar } from "@/components/chat/ChatInputBar";
import { useChatSession } from "@/hooks/useChatSession";
import { useChatController } from "@/hooks/chat/useChatController";
import { useChatMutations } from "@/hooks/chat/useChatMutations";
import { useSessionUIState } from "@/hooks/chat/useSessionUIState";
import { useAppConfig } from "@/components/config-provider";
import { useToast } from "@/components/toaster";
import { ProcessedSourcesPanel } from "@/components/chat/ProcessedSourcesPanel";

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
  const { threadId, sessionId, clearChat } = useChatSession();
  return <ChatPageContent key={threadId} appConfig={appConfig} threadId={threadId} sessionId={sessionId} clearChat={clearChat} />;
}

type ChatPageContentProps = {
  appConfig: ReturnType<typeof useAppConfig>["config"];
  threadId: string;
  sessionId: string;
  clearChat: ReturnType<typeof useChatSession>["clearChat"];
};

function ChatPageContent({ appConfig, threadId, sessionId, clearChat }: ChatPageContentProps) {
  const { toast } = useToast();
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
        onUpload={mutations.handleUpload}
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
              status={chat.status}
              referencesByAssistantIndex={chat.referencesByAssistantIndex}
              maxCitationsToShow={chat.maxCitationsToShow}
              chatContainerRef={chat.chatContainerRef}
              onRetry={chat.handleRetry}
              onFeedback={chat.handleFeedback}
              feedbackSubmitted={chat.feedbackSubmitted}
              enableUserFeedback={appConfig?.enable_user_feedback}
              pendingSuggestion={chat.pendingSuggestion}
              showOptimisticSuggestion={chat.showOptimisticSuggestion}
            />
            <ChatInputBar
              input={chat.input}
              setInput={chat.setInput}
              onSubmit={chat.handleSubmit}
              status={chat.status}
              dynamicSuggestions={chat.dynamicSuggestions}
              suggestionsLoading={chat.suggestionsLoading}
              pendingSuggestion={chat.pendingSuggestion}
              onSuggestionClick={chat.handleSuggestionClick}
            />
          </>
        ) : (
          <ProcessedSourcesPanel collectionName={sessionUI.collectionName} />
        )}
      </div>
    </div>
  );
}

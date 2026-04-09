"use client";

import Image from "next/image";
import { CheckIcon, PanelLeft, PanelLeftClose } from "lucide-react";
import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorEmpty,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorLogo,
  ModelSelectorLogoGroup,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "@/components/ai-elements/model-selector";
import { ContextUsageBadge } from "@/components/chat/ContextUsageBadge";

type ModelItem = {
  id: string;
  name: string;
  chefSlug: string;
  providers: string[];
};

type ContextUsage = {
  tokens: number;
  max: number;
  percent: number;
  model_id?: string;
};

type ChatHeaderProps = {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  modelList: ModelItem[];
  selectedModel: string;
  onSelectModel: (id: string) => void;
  modelSelectorOpen: boolean;
  onModelSelectorOpenChange: (open: boolean) => void;
  contextUsage: ContextUsage | null;
  selectedModelData: ModelItem | undefined;
};

export function ChatHeader({
  sidebarOpen,
  onToggleSidebar,
  modelList,
  selectedModel,
  onSelectModel,
  modelSelectorOpen,
  onModelSelectorOpenChange,
  contextUsage,
  selectedModelData,
}: ChatHeaderProps): React.ReactElement {
  return (
    <header className="flex w-full shrink-0 items-start justify-between gap-3 border-b border-border bg-card px-4 py-3 shadow-sm sm:items-center sm:gap-4 sm:px-6">
      <div className="flex min-w-0 flex-1 items-start gap-3 sm:items-center">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="mt-0.5 rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring sm:mt-0"
          aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
        >
          {sidebarOpen ? (
            <PanelLeftClose className="size-5" />
          ) : (
            <PanelLeft className="size-5" />
          )}
        </button>
        <div className="flex min-w-0 flex-1 flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
          <Image
            src="/oracle-logo.png"
            alt="Oracle"
            width={140}
            height={32}
            className="h-7 w-auto shrink-0 object-contain sm:h-8"
            priority
          />
          <div className="min-w-0 space-y-0.5">
            <h1 className="truncate text-lg font-semibold tracking-tight text-foreground sm:text-xl">
              OCI Custom RAG Agent
            </h1>
            <p className="truncate text-xs text-muted-foreground sm:text-sm">
              Powered by Oracle Cloud Infrastructure Generative AI with RAG
            </p>
          </div>
          {contextUsage != null ? (
            <ContextUsageBadge contextUsage={contextUsage} />
          ) : null}
        </div>
      </div>

      <ModelSelector
        open={modelSelectorOpen}
        onOpenChange={onModelSelectorOpenChange}
      >
        <ModelSelectorTrigger className="min-w-0 max-w-[12rem] rounded-md border border-input bg-background px-3 py-2 text-sm font-medium transition-colors hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-ring sm:max-w-[15rem] sm:min-w-[12rem]">
          <span className="flex min-w-0 items-center gap-2">
            {selectedModelData?.chefSlug ? (
              <ModelSelectorLogo
                provider={selectedModelData.chefSlug}
                className="shrink-0"
              />
            ) : null}
            <ModelSelectorName>
              {selectedModelData?.name || selectedModel}
            </ModelSelectorName>
          </span>
        </ModelSelectorTrigger>
        <ModelSelectorContent>
          <ModelSelectorInput placeholder="Search models..." />
          <ModelSelectorList>
            <ModelSelectorEmpty>No models found.</ModelSelectorEmpty>
            {modelList.map((m) => (
              <ModelSelectorItem
                key={m.id}
                onSelect={() => {
                  onSelectModel(m.id);
                  onModelSelectorOpenChange(false);
                }}
                value={m.id}
              >
                <ModelSelectorLogo provider={m.chefSlug} />
                <ModelSelectorName>{m.name}</ModelSelectorName>
                <ModelSelectorLogoGroup>
                  {m.providers.map((provider) => (
                    <ModelSelectorLogo key={provider} provider={provider} />
                  ))}
                </ModelSelectorLogoGroup>
                {selectedModel === m.id ? (
                  <CheckIcon className="ml-auto size-4" />
                ) : (
                  <div className="ml-auto size-4" />
                )}
              </ModelSelectorItem>
            ))}
          </ModelSelectorList>
        </ModelSelectorContent>
      </ModelSelector>
    </header>
  );
}

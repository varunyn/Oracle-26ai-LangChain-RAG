import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import {
  DEFAULT_MODEL_STORAGE_KEY,
  DEFAULT_MODELS,
} from "@/constants/chat";
import type { AppConfig } from "@/lib/config";

type FlowMode = "rag" | "mcp" | "mixed" | "direct";

function isValidFlowMode(value: unknown): value is FlowMode {
  return value === "rag" || value === "mcp" || value === "mixed" || value === "direct";
}

function getInitialFlowMode(): FlowMode {
  const defaultFlowEnv = process.env.NEXT_PUBLIC_DEFAULT_FLOW_MODE;
  return isValidFlowMode(defaultFlowEnv) ? defaultFlowEnv : "rag";
}

export function useSessionUIState(appConfig: AppConfig | null) {
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [collectionName, setCollectionName] = useState(
    () => appConfig?.collection_list?.[0] ?? "",
  );
  const [enableReranker, setEnableReranker] = useState(false);
  const [enableTracing, setEnableTracing] = useState(false);
  const [flowMode, setFlowMode] = useState<FlowMode>(getInitialFlowMode);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const hydrated = useSyncExternalStore(
    () => () => undefined,
    () => true,
    () => false,
  );

  const modelList = useMemo(
    () =>
      (appConfig?.model_list?.length
        ? appConfig.model_list.map((id: string) => ({
            id,
            name: appConfig.model_display_names?.[id] ?? id,
            chef: "OCI",
            chefSlug: "oci",
            providers: ["oci"],
          }))
        : DEFAULT_MODELS) as typeof DEFAULT_MODELS,
    [appConfig],
  );

  const collectionList = appConfig?.collection_list ?? ["DOCUMENT_CHUNKS_VS"];

  const persistedSelectedModel = useMemo(() => {
    if (!hydrated || typeof window === "undefined") return "";

    const stored = localStorage.getItem(DEFAULT_MODEL_STORAGE_KEY);
    return typeof stored === "string" ? stored : "";
  }, [hydrated]);

  const effectiveSelectedModel = useMemo(() => {
    if (!modelList.length) return "";
    const ids = modelList.map((m) => m.id);
    if (selectedModel && ids.includes(selectedModel)) return selectedModel;
    if (persistedSelectedModel && ids.includes(persistedSelectedModel)) {
      return persistedSelectedModel;
    }
    return modelList[0].id;
  }, [modelList, persistedSelectedModel, selectedModel]);

  useEffect(() => {
    if (typeof window === "undefined" || !effectiveSelectedModel) return;
    localStorage.setItem(DEFAULT_MODEL_STORAGE_KEY, effectiveSelectedModel);
  }, [effectiveSelectedModel]);

  const handleSelectModel = useCallback((id: string) => {
    setSelectedModel(id);
    if (typeof window !== "undefined") {
      localStorage.setItem(DEFAULT_MODEL_STORAGE_KEY, id);
    }
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((open) => !open);
  }, []);

  const selectedModelData =
    modelList.find((m) => m.id === effectiveSelectedModel) ?? modelList[0];

  const stableSelectedModelData = hydrated
    ? selectedModelData
    : modelList[0];

  const stableSelectedModel = hydrated
    ? effectiveSelectedModel
    : modelList[0]?.id ?? "";

  return {
    modelSelectorOpen,
    setModelSelectorOpen,
    collectionName,
    setCollectionName,
    enableReranker,
    setEnableReranker,
    enableTracing,
    setEnableTracing,
    flowMode,
    setFlowMode,
    sidebarOpen,
    toggleSidebar,
    modelList,
    collectionList,
    effectiveSelectedModel: stableSelectedModel,
    handleSelectModel,
    selectedModelData: stableSelectedModelData,
  };
}

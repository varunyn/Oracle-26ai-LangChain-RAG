"use client";

import {
  createContext,
  useContext,
  type ReactNode,
} from "react";
import type { AppConfig } from "@/lib/config";

type AppConfigContextValue = {
  config: AppConfig | null;
};

const AppConfigContext = createContext<AppConfigContextValue | null>(null);

export function ConfigProvider({
  initialConfig,
  children,
}: {
  initialConfig: AppConfig | null;
  children: ReactNode;
}) {
  return (
    <AppConfigContext.Provider value={{ config: initialConfig }}>
      {children}
    </AppConfigContext.Provider>
  );
}

export function useAppConfig(): AppConfigContextValue {
  const value = useContext(AppConfigContext);
  if (value === null) {
    throw new Error("useAppConfig must be used within ConfigProvider");
  }
  return value;
}

import { cache } from "react";

import { toApiUrl } from "@/lib/api-base";

export type ObservabilityLink = {
  key: string;
  label: string;
  enabled: boolean;
  configured: boolean;
  url?: string | null;
  status: string;
  details: string;
};

export type AppConfig = {
  region?: string;
  embed_model_id?: string;
  model_list?: string[];
  model_display_names?: Record<string, string>;
  collection_list?: string[];
  enable_user_feedback?: boolean;
  observability?: {
    links: ObservabilityLink[];
  };
};

const API_BASE_URL =
  process.env.FASTAPI_BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:3002";

/**
 * Fetch app config from the FastAPI backend. Not cached so .env changes
 * are reflected after backend restart and a refresh. Still cached per request in RSC.
 */
export const getAppConfig = cache(async (): Promise<AppConfig | null> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/config`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data as AppConfig;
  } catch {
    return null;
  }
});

export async function fetchAppConfig(): Promise<AppConfig> {
  const response = await fetch(toApiUrl("/api/config"), {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load app config");
  }
  return (await response.json()) as AppConfig;
}

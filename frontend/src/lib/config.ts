import { cache } from "react";

export type AppConfig = {
  region?: string;
  embed_model_id?: string;
  model_list?: string[];
  model_display_names?: Record<string, string>;
  collection_list?: string[];
  enable_user_feedback?: boolean;
};

const FASTAPI_BACKEND_URL =
  process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

/**
 * Fetch app config from the FastAPI backend. Not cached so .env changes
 * are reflected after backend restart and a refresh. Still cached per request in RSC.
 */
export const getAppConfig = cache(async (): Promise<AppConfig | null> => {
  try {
    const res = await fetch(`${FASTAPI_BACKEND_URL}/api/config`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data as AppConfig;
  } catch {
    return null;
  }
});

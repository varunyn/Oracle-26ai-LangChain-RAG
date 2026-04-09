/**
 * Chat constants and configurations
 * Extracted from page.tsx for better code organization
 */

/** localStorage key for persisting thread ID */
export const THREAD_ID_STORAGE_KEY = "rag_agent_thread_id";

/** localStorage key for user's default model (persists across refresh and server restarts) */
export const DEFAULT_MODEL_STORAGE_KEY = "rag_default_model";

// Citation markers [1], [2] in the response. We replace runs (e.g. "[1] [2] [3] [5]") with
// a single span so one pill shows the source filename and hover shows a carousel of all cited chunks.
export const CITATION_MARKER_REGEX = /\[(\d+)\]/g;

/** Matches a run of citation markers so we can replace with one pill (e.g. "[1] [2] [3] [5]"). */
export const CITATION_RUN_REGEX = /\[\d+\](?:\s*\[\d+\])*/g;

/** Suggested questions shown on welcome screen */
export const SUGGESTIONS = [
  "Summarize the skills and experience across the uploaded resumes.",
  "Which documents mention Oracle Cloud Infrastructure setup or access?",
  "List the main policies or procedures referenced in this collection.",
];

/** Default model configurations */
export const DEFAULT_MODELS = [
  {
    id: "meta.llama-3.3-70b-instruct",
    name: "Llama 3.3 70B",
    chef: "Meta",
    chefSlug: "llama",
    providers: ["oci"],
  },
  {
    id: "xai.grok-4-fast-reasoning",
    name: "Grok 4 Fast (Reasoning)",
    chef: "xAI",
    chefSlug: "xai",
    providers: ["oci"],
  },
];

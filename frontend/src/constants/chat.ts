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
  "Tell me about Oracle 26ai Database.",
  "Solve this math problem: 125 * 48.",
  "What can you help me find in my documents?",
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

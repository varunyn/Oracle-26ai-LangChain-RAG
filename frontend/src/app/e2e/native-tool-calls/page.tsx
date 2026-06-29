"use client";

import type { AssembledToolCall } from "@langchain/react";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { useSyncExternalStore } from "react";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { projectStreamMessages } from "@/hooks/chat/message-projection";

type Phase = "running" | "final";

const STORAGE_KEY = "native-tool-calls-e2e-phase";
const PHASE_EVENT = "native-tool-calls-e2e-phase-change";

const PRIOR_USER_MESSAGES = Array.from({ length: 4 }, (_, index) =>
  new HumanMessage({
    id: `history-user-${index + 1}`,
    content: `Historical user context ${index + 1}: Summarize the invoice review notes for batch ${
      index + 1
    } and keep the unresolved exceptions visible for finance follow-up.`,
  }),
);

const RUNNING_STREAM_MESSAGES = [
  ...PRIOR_USER_MESSAGES,
  new HumanMessage({
    id: "user-1",
    content: "Audit invoice exceptions with multiple native tool calls",
  }),
  new AIMessage({
    id: "assistant-1",
    content: "",
    tool_calls: [
      {
        id: "calc-1",
        name: "calculate_invoice_total",
        args: { invoiceId: "INV-42", lineItems: [125, 250.5] },
        type: "tool_call",
      },
      {
        id: "lookup-1",
        name: "lookup_vendor_profile",
        args: { vendorId: "northwell-001", includeInvoices: true },
        type: "tool_call",
      },
    ],
  }),
];

const FINAL_STREAM_MESSAGES = [
  ...PRIOR_USER_MESSAGES,
  new HumanMessage({
    id: "user-1",
    content: "Audit invoice exceptions with multiple native tool calls",
  }),
  new AIMessage({
    id: "assistant-1",
    content: "I checked the invoice totals and requested the vendor lookup.",
    tool_calls: [
      {
        id: "calc-1",
        name: "calculate_invoice_total",
        args: { invoiceId: "INV-42", lineItems: [125, 250.5] },
        type: "tool_call",
      },
      {
        id: "lookup-1",
        name: "lookup_vendor_profile",
        args: { vendorId: "northwell-001", includeInvoices: true },
        type: "tool_call",
      },
    ],
    additional_kwargs: {
      citations: [],
      reranker_docs: [],
    },
  }),
  new AIMessage({
    id: "assistant-2",
    content: "The vendor lookup failed, so I used the fallback summary instead.",
    tool_calls: [
      {
        id: "summary-1",
        name: "summarize_invoice_risk",
        args: { invoiceId: "INV-42", confidence: 0.82 },
        type: "tool_call",
      },
    ],
    additional_kwargs: {
      citations: [],
      reranker_docs: [],
    },
  }),
];

const RUNNING_TOOL_CALLS = [
  {
    callId: "calc-1",
    id: "calc-1",
    name: "calculate_invoice_total",
    namespace: [],
    input: { invoiceId: "INV-42", lineItems: [125, 250.5] },
    args: { invoiceId: "INV-42", lineItems: [125, 250.5] },
    output: null,
    status: "running",
    error: undefined,
  },
  {
    callId: "lookup-1",
    id: "lookup-1",
    name: "lookup_vendor_profile",
    namespace: [],
    input: { vendorId: "northwell-001", includeInvoices: true },
    args: { vendorId: "northwell-001", includeInvoices: true },
    output: null,
    status: "running",
    error: undefined,
  },
] satisfies AssembledToolCall[];

const FINAL_TOOL_CALLS = [
  {
    callId: "calc-1",
    id: "calc-1",
    name: "calculate_invoice_total",
    namespace: [],
    input: { invoiceId: "INV-42", lineItems: [125, 250.5] },
    args: { invoiceId: "INV-42", lineItems: [125, 250.5] },
    output: { total: 375.5, currency: "USD" },
    status: "finished",
    error: undefined,
  },
  {
    callId: "lookup-1",
    id: "lookup-1",
    name: "lookup_vendor_profile",
    namespace: [],
    input: { vendorId: "northwell-001", includeInvoices: true },
    args: { vendorId: "northwell-001", includeInvoices: true },
    output: null,
    status: "error",
    error: "Vendor service timeout",
  },
  {
    callId: "summary-1",
    id: "summary-1",
    name: "summarize_invoice_risk",
    namespace: [],
    input: { invoiceId: "INV-42", confidence: 0.82 },
    args: { invoiceId: "INV-42", confidence: 0.82 },
    output: { risk: "medium", reason: "Vendor profile unavailable" },
    status: "finished",
    error: undefined,
  },
] satisfies AssembledToolCall[];

function readPhaseSnapshot(): Phase {
  if (typeof window === "undefined") return "running";
  return window.localStorage.getItem(STORAGE_KEY) === "final" ? "final" : "running";
}

function subscribeToPhase(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};

  const handleChange = () => {
    onStoreChange();
  };

  window.addEventListener("storage", handleChange);
  window.addEventListener(PHASE_EVENT, handleChange);

  return () => {
    window.removeEventListener("storage", handleChange);
    window.removeEventListener(PHASE_EVENT, handleChange);
  };
}

function setStoredPhase(phase: Phase): void {
  window.localStorage.setItem(STORAGE_KEY, phase);
  window.dispatchEvent(new Event(PHASE_EVENT));
}

function phaseFixture(phase: Phase) {
  return {
    messages:
      phase === "final"
        ? projectStreamMessages({ streamMessages: [...FINAL_STREAM_MESSAGES] })
        : projectStreamMessages({ streamMessages: [...RUNNING_STREAM_MESSAGES] }),
    toolCalls: phase === "final" ? [...FINAL_TOOL_CALLS] : [...RUNNING_TOOL_CALLS],
    status: phase === "final" ? "ready" : "streaming",
  };
}

function useStoredPhase(): Phase {
  return useSyncExternalStore(subscribeToPhase, readPhaseSnapshot, () => "running");
}

export default function NativeToolCallsE2EPage(): React.ReactElement {
  const phase = useStoredPhase();
  const fixture = phaseFixture(phase);

  return (
    <main className="min-h-screen bg-background px-6 py-8" data-testid="native-tool-calls-e2e">
      <div className="mx-auto flex max-w-5xl flex-col gap-4">
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => {
              setStoredPhase("running");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Reset Running
          </button>
          <button
            type="button"
            onClick={() => {
              setStoredPhase("final");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Advance Final
          </button>
          <button
            type="button"
            onClick={() => {
              setStoredPhase("final");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Replay Final Snapshot
          </button>
        </div>

        <div className="flex h-[260px] min-h-0 rounded-xl border bg-card">
          <ChatMessageList
            messages={fixture.messages}
            toolCalls={fixture.toolCalls}
            status={fixture.status}
            maxCitationsToShow={10}
            onRetry={() => {}}
            onRecoverDirect={() => {}}
            onRecoverRagOnly={() => {}}
            onFeedback={() => {}}
            feedbackSubmittedMessageIndexes={new Set<number>()}
            enableUserFeedback={false}
          />
        </div>
      </div>
    </main>
  );
}

"use client";

import type { AssembledToolCall } from "@langchain/react";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { useEffect, useRef, useState } from "react";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { projectStreamMessages } from "@/hooks/chat/message-projection";

type Phase = "running" | "final";

const STORAGE_KEY = "native-tool-calls-e2e-phase";

const RUNNING_STREAM_MESSAGES = [
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

function readInitialPhase(): Phase {
  if (typeof window === "undefined") return "running";
  return window.localStorage.getItem(STORAGE_KEY) === "final" ? "final" : "running";
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

export default function NativeToolCallsE2EPage(): React.ReactElement {
  const [phase, setPhase] = useState<Phase>("running");
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setPhase(readInitialPhase());
  }, []);

  const fixture = phaseFixture(phase);

  return (
    <main className="min-h-screen bg-background px-6 py-8" data-testid="native-tool-calls-e2e">
      <div className="mx-auto flex max-w-5xl flex-col gap-4">
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => {
              window.localStorage.setItem(STORAGE_KEY, "running");
              setPhase("running");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Reset Running
          </button>
          <button
            type="button"
            onClick={() => {
              window.localStorage.setItem(STORAGE_KEY, "final");
              setPhase("final");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Advance Final
          </button>
          <button
            type="button"
            onClick={() => {
              window.localStorage.setItem(STORAGE_KEY, "final");
              setPhase("final");
            }}
            className="rounded border px-3 py-2 text-sm"
          >
            Replay Final Snapshot
          </button>
        </div>

        <div className="rounded-xl border bg-card">
          <ChatMessageList
            messages={fixture.messages}
            toolCalls={fixture.toolCalls}
            status={fixture.status}
            maxCitationsToShow={10}
            chatContainerRef={chatContainerRef}
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

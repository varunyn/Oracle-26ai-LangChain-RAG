"use client";

import * as React from "react";

import {
  ToastProvider,
  ToastViewport,
  ToastRoot,
  ToastTitle,
  ToastDescription,
  ToastClose,
  toastVariants,
} from "@/components/ui/toast";

type ToastState = {
  open: boolean;
  title: string;
  description: string;
  variant: "default" | "destructive";
};

type ToastContextValue = {
  toast: {
    error: (description: string, title?: string) => void;
    success: (description: string, title?: string) => void;
  };
};

const ToastContext = React.createContext<ToastContextValue | null>(null);

const TOAST_DURATION = 8_000;

export function useToast() {
  const ctx = React.useContext(ToastContext);
  if (!ctx) {
    return {
      toast: {
        error: (
          description: string,
          title = "Response unavailable",
        ) => {
          if (typeof console !== "undefined" && console.error) {
            console.error(
              "[Toast] useToast must be used within ToasterProvider:",
              `${title}: ${description}`,
            );
          }
        },
        success: () => {},
      },
    };
  }
  return ctx;
}

export function ToasterProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<ToastState>({
    open: false,
    title: "Notice",
    description: "",
    variant: "default",
  });

  const toast = React.useMemo(
    () => ({
      error: (description: string, title = "Response unavailable") => {
        setState({
          open: true,
          title,
          description,
          variant: "destructive",
        });
      },
      success: (description: string, title = "Update complete") => {
        setState({
          open: true,
          title,
          description,
          variant: "default",
        });
      },
    }),
    [],
  );

  return (
    <ToastContext.Provider value={{ toast }}>
      <ToastProvider duration={TOAST_DURATION} swipeDirection="right">
        {children}
        <ToastViewport />
        <ToastRoot
          open={state.open}
          onOpenChange={(open) => setState((prev) => ({ ...prev, open }))}
          duration={TOAST_DURATION}
          className={toastVariants[state.variant]}
          type="foreground"
        >
          <ToastTitle>{state.title}</ToastTitle>
          <ToastDescription>{state.description}</ToastDescription>
          <ToastClose aria-label="Close" />
        </ToastRoot>
      </ToastProvider>
    </ToastContext.Provider>
  );
}

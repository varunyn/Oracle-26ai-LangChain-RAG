"use client";

import { ChevronDown } from "lucide-react";
import * as React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const SCROLL_BOTTOM_THRESHOLD = 24;

type ConversationContextValue = {
  contentRef: React.RefObject<HTMLDivElement | null>;
  showScrollButton: boolean;
  scrollToBottom: (behavior?: ScrollBehavior) => void;
};

const ConversationContext =
  React.createContext<ConversationContextValue | null>(null);

function useConversationContext(): ConversationContextValue {
  const context = React.useContext(ConversationContext);
  if (context == null) {
    throw new Error(
      "Conversation components must be used within <Conversation>."
    );
  }
  return context;
}

function composeRefs<T>(
  ...refs: Array<React.Ref<T> | undefined>
): (node: T | null) => void {
  return (node) => {
    for (const ref of refs) {
      if (typeof ref === "function") {
        ref(node);
      } else if (ref != null) {
        (ref as React.MutableRefObject<T | null>).current = node;
      }
    }
  };
}

export const Conversation = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function Conversation({ children, className, ...props }, forwardedRef) {
  const viewportRef = React.useRef<HTMLDivElement | null>(null);
  const contentRef = React.useRef<HTMLDivElement | null>(null);
  const isAtBottomRef = React.useRef(true);
  const didInitialScrollRef = React.useRef(false);
  const [showScrollButton, setShowScrollButton] = React.useState(false);

  const updateScrollState = React.useCallback(() => {
    const viewport = viewportRef.current;
    if (viewport == null) {
      return;
    }

    const distanceFromBottom =
      viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop;
    const isAtBottom = distanceFromBottom <= SCROLL_BOTTOM_THRESHOLD;
    const hasOverflow = viewport.scrollHeight > viewport.clientHeight + 1;

    isAtBottomRef.current = isAtBottom;
    setShowScrollButton(hasOverflow && !isAtBottom);
  }, []);

  const scrollToBottom = React.useCallback(
    (behavior: ScrollBehavior = "auto") => {
      const viewport = viewportRef.current;
      if (viewport == null) {
        return;
      }
      viewport.scrollTo({ top: viewport.scrollHeight, behavior });
      isAtBottomRef.current = true;
      setShowScrollButton(false);
    },
    []
  );

  React.useEffect(() => {
    const viewport = viewportRef.current;
    if (viewport == null) {
      return;
    }

    const handleScroll = () => {
      updateScrollState();
    };

    viewport.addEventListener("scroll", handleScroll, { passive: true });
    updateScrollState();

    return () => {
      viewport.removeEventListener("scroll", handleScroll);
    };
  }, [updateScrollState]);

  React.useEffect(() => {
    const viewport = viewportRef.current;
    const content = contentRef.current;
    if (viewport == null || content == null) {
      return;
    }

    const syncAfterContentChange = () => {
      const shouldStickToBottom =
        !didInitialScrollRef.current || isAtBottomRef.current;
      updateScrollState();
      if (shouldStickToBottom) {
        scrollToBottom("auto");
      }
      didInitialScrollRef.current = true;
      updateScrollState();
    };

    const resizeObserver = new ResizeObserver(() => {
      syncAfterContentChange();
    });
    const mutationObserver = new MutationObserver(() => {
      syncAfterContentChange();
    });

    resizeObserver.observe(content);
    mutationObserver.observe(content, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    requestAnimationFrame(() => {
      syncAfterContentChange();
    });

    return () => {
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [scrollToBottom, updateScrollState]);

  return (
    <ConversationContext.Provider
      value={{ contentRef, showScrollButton, scrollToBottom }}
    >
      <div
        className={cn(
          "relative flex min-h-0 w-full flex-1 flex-col overflow-y-auto overflow-x-hidden overscroll-contain",
          className
        )}
        ref={composeRefs(forwardedRef, viewportRef)}
        {...props}
      >
        {children}
      </div>
    </ConversationContext.Provider>
  );
});

export const ConversationContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function ConversationContent(
  { children, className, ...props },
  forwardedRef
) {
  const { contentRef } = useConversationContext();

  return (
    <div
      className={cn("flex w-full flex-col", className)}
      ref={composeRefs(forwardedRef, contentRef)}
      {...props}
    >
      {children}
    </div>
  );
});

export const ConversationScrollButton = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<typeof Button>
>(function ConversationScrollButton(
  { className, onClick, type = "button", ...props },
  forwardedRef
) {
  const { scrollToBottom, showScrollButton } = useConversationContext();

  return (
    <Button
      aria-label="Scroll to latest"
      className={cn(
        "absolute right-4 bottom-4 z-10 rounded-full border shadow-sm",
        className
      )}
      data-testid="chat-scroll-to-bottom"
      hidden={!showScrollButton}
      onClick={(event) => {
        onClick?.(event);
        if (!event.defaultPrevented) {
          scrollToBottom();
        }
      }}
      ref={forwardedRef}
      size="icon"
      type={type}
      variant="secondary"
      {...props}
    >
      <ChevronDown className="size-4" />
      <span className="sr-only">Scroll to latest</span>
    </Button>
  );
});

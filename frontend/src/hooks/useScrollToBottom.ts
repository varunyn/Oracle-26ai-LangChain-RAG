import { useEffect, useRef } from "react";

/**
 * Provides a ref for the chat scroll container and scrolls to bottom when
 * status/messages change (unless the user has scrolled up). Uses passive scroll listener.
 */
export function useScrollToBottom<T>(status: string, messages: T[]): React.RefObject<HTMLDivElement | null> {
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);

  useEffect(() => {
    const el = chatContainerRef.current;
    if (el == null) return;
    const onScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el;
      userScrolledUpRef.current = scrollTop < scrollHeight - clientHeight - 50;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (status !== "submitted" && status !== "streaming") return;
    if (userScrolledUpRef.current) return;
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [status, messages]);

  return chatContainerRef;
}

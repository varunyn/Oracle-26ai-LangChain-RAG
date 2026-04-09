"use client";

import type { CarouselApi } from "@/components/ui/carousel";
import type { ComponentProps, ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";
import { ArrowLeftIcon, ArrowRightIcon } from "lucide-react";
import {
  Children,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useReducer,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type InlineCitationProps = ComponentProps<"span">;

export const InlineCitation = ({
  className,
  ...props
}: InlineCitationProps) => (
  <span
    className={cn("group inline items-center gap-1", className)}
    {...props}
  />
);

type InlineCitationTextProps = ComponentProps<"span">;

export const InlineCitationText = ({
  className,
  ...props
}: InlineCitationTextProps) => (
  <span
    className={cn("transition-colors group-hover:bg-accent", className)}
    {...props}
  />
);

export type InlineCitationCardProps = ComponentProps<typeof HoverCard>;

export const InlineCitationCard = (props: InlineCitationCardProps) => (
  <HoverCard closeDelay={0} openDelay={0} {...props} />
);

function getSourceLabel(source: string): string {
  try {
    if (source.startsWith("file://") || source.includes("/")) {
      const parts = source.split("/");
      return parts[parts.length - 1] ?? source;
    }
    return new URL(source).hostname;
  } catch {
    return source;
  }
}

export type InlineCitationCardTriggerProps = ComponentProps<typeof Badge> & {
  sources: string[];
  /** Optional compact label (e.g. "[1]" or "Source (×3)"). When set, badge shows this instead of source. */
  label?: string;
};

export const InlineCitationCardTrigger = ({
  sources,
  label,
  className,
  children,
  ...props
}: InlineCitationCardTriggerProps) => (
  <HoverCardTrigger asChild>
    <Badge
      className={cn("ml-1 rounded-full", className)}
      variant="secondary"
      {...props}
    >
      {children ??
        (label != null
          ? label
          : sources[0]
            ? (() => {
                const first = getSourceLabel(sources[0]);
                return (
                  <>
                    {first}
                    {sources.length > 1 && ` +${sources.length - 1}`}
                  </>
                );
              })()
            : "unknown")}
    </Badge>
  </HoverCardTrigger>
);

export type InlineCitationCardBodyProps = ComponentProps<"div">;

export const InlineCitationCardBody = ({
  className,
  ...props
}: InlineCitationCardBodyProps) => (
  <HoverCardContent className={cn("relative w-80 p-0", className)} {...props} />
);

const CarouselApiContext = createContext<CarouselApi | undefined>(undefined);

const useCarouselApi = () => {
  const context = useContext(CarouselApiContext);
  return context;
};

export type InlineCitationCarouselProps = ComponentProps<typeof Carousel>;

export const InlineCitationCarousel = ({
  className,
  children,
  ...props
}: InlineCitationCarouselProps) => {
  const [api, setApi] = useState<CarouselApi>();

  return (
    <CarouselApiContext.Provider value={api}>
      <Carousel className={cn("w-full", className)} setApi={setApi} {...props}>
        {children}
      </Carousel>
    </CarouselApiContext.Provider>
  );
};

export type InlineCitationCarouselContentProps = ComponentProps<"div">;

export const InlineCitationCarouselContent = (
  props: InlineCitationCarouselContentProps
) => <CarouselContent {...props} />;

export type InlineCitationCarouselItemProps = ComponentProps<"div">;

export const InlineCitationCarouselItem = ({
  className,
  ...props
}: InlineCitationCarouselItemProps) => (
  <CarouselItem
    className={cn("w-full space-y-2 p-4 pl-8", className)}
    {...props}
  />
);

export type InlineCitationCarouselHeaderProps = ComponentProps<"div">;

export const InlineCitationCarouselHeader = ({
  className,
  ...props
}: InlineCitationCarouselHeaderProps) => (
  <div
    className={cn(
      "flex items-center justify-between gap-2 rounded-t-md bg-secondary p-2",
      className
    )}
    {...props}
  />
);

export type InlineCitationCarouselIndexProps = ComponentProps<"div">;

export const InlineCitationCarouselIndex = ({
  children,
  className,
  ...props
}: InlineCitationCarouselIndexProps) => {
  const api = useCarouselApi();
  const [, forceRender] = useReducer((v: number) => v + 1, 0);

  const current = api ? api.selectedScrollSnap() + 1 : 0;
  const count = api ? api.scrollSnapList().length : 0;

  useEffect(() => {
    if (!api) {
      return;
    }

    const handleSelect = () => {
      forceRender();
    };

    api.on("select", handleSelect);

    return () => {
      api.off("select", handleSelect);
    };
  }, [api, forceRender]);

  return (
    <div
      className={cn(
        "flex flex-1 items-center justify-end px-3 py-1 text-muted-foreground text-xs",
        className
      )}
      {...props}
    >
      {children ?? `${current}/${count}`}
    </div>
  );
};

export type InlineCitationCarouselPrevProps = ComponentProps<"button">;

export const InlineCitationCarouselPrev = ({
  className,
  ...props
}: InlineCitationCarouselPrevProps) => {
  const api = useCarouselApi();

  const handleClick = useCallback(() => {
    if (api) {
      api.scrollPrev();
    }
  }, [api]);

  return (
    <button
      aria-label="Previous"
      className={cn("shrink-0", className)}
      onClick={handleClick}
      type="button"
      {...props}
    >
      <ArrowLeftIcon className="size-4 text-muted-foreground" />
    </button>
  );
};

export type InlineCitationCarouselNextProps = ComponentProps<"button">;

export const InlineCitationCarouselNext = ({
  className,
  ...props
}: InlineCitationCarouselNextProps) => {
  const api = useCarouselApi();

  const handleClick = useCallback(() => {
    if (api) {
      api.scrollNext();
    }
  }, [api]);

  return (
    <button
      aria-label="Next"
      className={cn("shrink-0", className)}
      onClick={handleClick}
      type="button"
      {...props}
    >
      <ArrowRightIcon className="size-4 text-muted-foreground" />
    </button>
  );
};

export type InlineCitationSourceProps = ComponentProps<"div"> & {
  title?: string;
  url?: string;
  description?: string;
};

export const InlineCitationSource = ({
  title,
  url,
  description,
  className,
  children,
  ...props
}: InlineCitationSourceProps) => (
  <div className={cn("space-y-1", className)} {...props}>
    {title && (
      <h4 className="truncate font-medium text-sm leading-tight">{title}</h4>
    )}
    {url && (
      <p className="truncate break-all text-muted-foreground text-xs">{url}</p>
    )}
    {description && (
      <p className="line-clamp-3 text-muted-foreground text-sm leading-relaxed">
        {description}
      </p>
    )}
    {children}
  </div>
);

/** Normalize React children to a single markdown string (e.g. multiple text nodes from JSX). */
function childrenToMarkdown(node: ReactNode): string {
  if (typeof node === "string") return node;
  const arr = Children.toArray(node);
  return arr
    .map((c) => (typeof c === "string" ? c : ""))
    .join("")
    .trim();
}

const citationQuoteStyles =
  "text-foreground/90 text-sm [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_pre]:text-xs [&_pre]:py-1.5 [&_pre]:px-2 [&_pre]:rounded [&_pre]:bg-muted/50 [&_code]:text-xs [&_ul]:my-1 [&_ol]:my-1 [&_p]:my-0.5 [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-sm";

export type InlineCitationQuoteProps = ComponentProps<"blockquote"> & {
  label?: string;
};

export const InlineCitationQuote = ({
  children,
  label,
  className,
  ...props
}: InlineCitationQuoteProps) => {
  const markdown = childrenToMarkdown(children);
  const content =
    markdown.length > 0 ? (
      <div className={cn(citationQuoteStyles, className)}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
      </div>
    ) : (
      <span className={cn("text-sm text-foreground/90", className)}>
        {children as ReactNode}
      </span>
    );
  return (
    <blockquote
      className={cn(
        "border-muted border-l-2 pl-3 text-muted-foreground not-italic max-h-40 overflow-y-auto overflow-x-hidden",
        label != null && "mt-1"
      )}
      {...props}
    >
      {label ? (
        <p className="text-muted-foreground text-xs font-medium uppercase tracking-wide mb-1">
          {label}
        </p>
      ) : null}
      {content}
    </blockquote>
  );
};

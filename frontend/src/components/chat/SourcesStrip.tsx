"use client";

import { memo } from "react";
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from "@/components/ai-elements/sources";

type CitationRef = { source: string; page: string | null; link?: string | null };

interface SourcesStripProps {
  citations: CitationRef[];
  maxToShow?: number;
}

function sourceTitle(source: string): string {
  try {
    const url = new URL(source);
    return url.hostname;
  } catch {
    return source.split("/").pop() || source;
  }
}

function sourceHref(citation: CitationRef): string | undefined {
  const href = citation.link || citation.source;
  if (href.startsWith("http://") || href.startsWith("https://")) {
    return href;
  }
  return undefined;
}

function uniqueSourcesWithCount(
  citations: CitationRef[],
  maxToShow: number,
): {
  count: number;
  firstIndex: number;
  link?: string | null;
  page: string;
  source: string;
}[] {
  const bySource = new Map<
    string,
    {
      count: number;
      firstIndex: number;
      link?: string | null;
      page: string;
      source: string;
    }
  >();

  citations.forEach((citation, index) => {
    const source = citation.source?.trim();
    if (!source) return;

    const existing = bySource.get(source);
    if (existing) {
      existing.count += 1;
      return;
    }

    bySource.set(source, {
      count: 1,
      firstIndex: index,
      link: citation.link,
      page: citation.page ?? "",
      source,
    });
  });

  return Array.from(bySource.values()).slice(0, maxToShow);
}

export const SourcesStrip = memo(function SourcesStrip({
  citations,
  maxToShow = 10,
}: SourcesStripProps) {
  const sources = uniqueSourcesWithCount(citations, maxToShow);
  if (sources.length === 0) return null;

  return (
    <Sources className="mt-3 border-t border-border/60 pt-2">
      <SourcesTrigger count={sources.length} />
      <SourcesContent>
        {sources.map((source) => {
          const details = [
            source.page ? `Page ${source.page}` : null,
            source.count > 1 ? `${source.count} chunks` : null,
          ].filter((detail): detail is string => detail != null);

          return (
            <Source
              description={details.join(" · ") || undefined}
              href={sourceHref(source)}
              key={`${source.source}-${source.firstIndex}`}
              title={sourceTitle(source.source)}
            />
          );
        })}
      </SourcesContent>
    </Sources>
  );
});

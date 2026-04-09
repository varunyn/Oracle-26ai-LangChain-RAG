"use client";

import { memo } from "react";
import {
  InlineCitation,
  InlineCitationCard,
  InlineCitationCardBody,
  InlineCitationCardTrigger,
  InlineCitationCarousel,
  InlineCitationCarouselContent,
  InlineCitationCarouselHeader,
  InlineCitationCarouselIndex,
  InlineCitationCarouselItem,
  InlineCitationCarouselNext,
  InlineCitationCarouselPrev,
  InlineCitationQuote,
  InlineCitationSource,
} from "@/components/ai-elements/inline-citation";

type CitationRef = { source: string; page: string };
type RerankerDoc = { page_content: string; metadata: Record<string, unknown> };

interface SourcesStripProps {
  citations: CitationRef[];
  rerankerDocs?: RerankerDoc[];
  maxToShow?: number;
}

/** Unique source entry for deduplicated strip (one pill per source, with count and all indices). */
function uniqueSourcesWithCount(
  citations: CitationRef[],
  maxToShow: number
): { source: string; page: string; count: number; firstIndex: number; allIndices: number[] }[] {
  const bySource = new Map<
    string,
    { page: string; count: number; firstIndex: number; allIndices: number[] }
  >();
  
  citations.forEach((c, i) => {
    const key = (c.source ?? "").trim() || `__empty_${i}`;
    if (!bySource.has(key)) {
      bySource.set(key, { page: c.page ?? "", count: 1, firstIndex: i, allIndices: [i] });
    } else {
      const entry = bySource.get(key)!;
      entry.count += 1;
      entry.allIndices.push(i);
    }
  });
  
  return Array.from(bySource.entries())
    .map(([source, { page, count, firstIndex, allIndices }]) => ({
      source,
      page,
      count,
      firstIndex,
      allIndices,
    }))
    .filter((u) => u.source && !u.source.startsWith("__empty"))
    .slice(0, maxToShow);
}

/**
 * Sources strip: one pill per unique source (deduplicated), with optional count to reduce clutter.
 * Extracted from page.tsx for better code organization.
 * React best practice: rerender-memo - memoized component for performance
 */
export const SourcesStrip = memo(function SourcesStrip({
  citations,
  rerankerDocs,
  maxToShow = 10,
}: SourcesStripProps) {
  if (citations.length === 0) return null;
  
  const unique = uniqueSourcesWithCount(citations, maxToShow);
  if (unique.length === 0) return null;
  
  return (
    <div className="mt-3 pt-2 border-t border-border/60">
      <span className="text-muted-foreground text-xs font-medium mr-2">
        Sources:
      </span>
      <span className="inline-flex flex-wrap gap-1.5 align-baseline">
        {unique.map((u) => {
          const sourceName = u.source?.split("/").pop() ?? "Source";
          const label = u.count > 1 ? `${sourceName} +${u.count - 1}` : sourceName;
          
          // If multiple citations from same source, show carousel
          if (u.count > 1 && u.allIndices && rerankerDocs) {
            return (
              <InlineCitation key={u.source}>
                <InlineCitationCard>
                  <InlineCitationCardTrigger
                    sources={[u.source]}
                    label={label}
                  />
                  <InlineCitationCardBody>
                    <InlineCitationCarousel>
                      <InlineCitationCarouselHeader>
                        <InlineCitationCarouselPrev />
                        <InlineCitationCarouselNext />
                        <InlineCitationCarouselIndex />
                      </InlineCitationCarouselHeader>
                      <InlineCitationCarouselContent>
                        {u.allIndices.map((idx) => {
                          const citation = citations[idx];
                          const doc = rerankerDocs[idx];
                          return (
                            <InlineCitationCarouselItem key={idx}>
                              <InlineCitationSource
                                title={sourceName}
                                url={citation?.source}
                                description={citation?.page ?? undefined}
                              />
                              {doc?.page_content ? (
                                <InlineCitationQuote>
                                  {doc.page_content.slice(0, 500)}
                                  {doc.page_content.length > 500 ? "…" : ""}
                                </InlineCitationQuote>
                              ) : null}
                            </InlineCitationCarouselItem>
                          );
                        })}
                      </InlineCitationCarouselContent>
                    </InlineCitationCarousel>
                  </InlineCitationCardBody>
                </InlineCitationCard>
              </InlineCitation>
            );
          }
          
          // Single citation - no carousel needed
          return (
            <InlineCitation key={u.source}>
              <InlineCitationCard>
                <InlineCitationCardTrigger
                  sources={[u.source]}
                  label={label}
                />
                <InlineCitationCardBody>
                  <InlineCitationSource
                    title={sourceName}
                    url={u.source}
                    description={u.page ? u.page : undefined}
                  />
                  {rerankerDocs?.[u.firstIndex]?.page_content ? (
                    <InlineCitationQuote>
                      {rerankerDocs[u.firstIndex].page_content.slice(0, 500)}
                      {rerankerDocs[u.firstIndex].page_content.length > 500
                        ? "…"
                        : ""}
                    </InlineCitationQuote>
                  ) : null}
                </InlineCitationCardBody>
              </InlineCitationCard>
            </InlineCitation>
          );
        })}
      </span>
    </div>
  );
});

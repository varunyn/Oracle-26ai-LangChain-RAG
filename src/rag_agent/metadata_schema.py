"""
Metadata schema for RAG chunks and citations.

Contract for ingestion (`src/rag_agent/ingestion.py`, API upload, CLI wrapper, etc.):
- Set `source_url` on each chunk as the preferred source identity for source management and deletion.
  Also set one or more display-friendly fallback keys such as `file_name`, `source`, or `file_path`.
- Set at least one page key when applicable: page_label, page (e.g. PDF page),
  or chunk_offset (0-based index) so citations can show "Chunk 1", "Chunk 2", etc.

Consumers (reranker.generate_refs) use get_source_from_metadata and get_page_from_metadata
so that citation pills show the right source and location.
"""

# Preferred keys for source/citation display. `source_url` is the primary source-management identity;
# file_name/source remain useful display fallbacks.
SOURCE_KEYS = ("source_url", "file_name", "source", "file_path")
PAGE_KEYS = ("page_label", "page", "chunk_offset")

# Defaults when metadata is missing
UNKNOWN_SOURCE = "Unknown"
EMPTY_PAGE = ""


def _normalize_metadata(metadata: dict) -> dict:
    """
    Normalize metadata keys to lowercase so we handle Oracle/drivers
    that return JSON keys in uppercase (e.g. SOURCE, FILE_NAME).
    """
    if not metadata or not isinstance(metadata, dict):
        return metadata or {}
    return {str(k).lower(): v for k, v in metadata.items()}


def get_source_from_metadata(metadata: dict) -> str:
    """
    Extract a source string from chunk metadata.
    Prefers source_url for stable identity, then file_name/source for display-oriented fallbacks.
    """
    if not metadata:
        return UNKNOWN_SOURCE
    meta = _normalize_metadata(metadata)
    for key in SOURCE_KEYS:
        val = meta.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return UNKNOWN_SOURCE


def get_page_from_metadata(metadata: dict) -> str:
    """
    Extract page or chunk offset from chunk metadata for citation display.
    """
    if not metadata:
        return EMPTY_PAGE
    meta = _normalize_metadata(metadata)
    for key in PAGE_KEYS:
        val = meta.get(key)
        if val is not None:
            s = str(val).strip() if not isinstance(val, (int, float)) else str(val)
            if s:
                # Normalize chunk_offset display (0-based -> "Chunk 1", "Chunk 2", ...)
                if key == "chunk_offset":
                    try:
                        n = int(float(s))
                        return f"Chunk {n + 1}"
                    except (ValueError, TypeError):
                        pass
                return s
    return EMPTY_PAGE

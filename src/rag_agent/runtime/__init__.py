"""Application runtime components used by the FastAPI API layer."""

from .agent import normalize_messages

__all__ = ["normalize_messages"]

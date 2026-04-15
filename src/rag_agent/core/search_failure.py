
from __future__ import annotations

SEARCH_ERROR_MESSAGE = (
    "Search is temporarily unavailable (e.g. database connection error). "
    "Please check your connection and try again later."
)


def search_error_response(state: dict[str, object]) -> dict[str, str]:
    err = state.get("error") or "Unknown error"
    return {"final_answer": f"{SEARCH_ERROR_MESSAGE}\n\nDetails: {err}"}

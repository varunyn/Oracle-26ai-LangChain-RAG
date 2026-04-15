"""Health check endpoint for the runtime API surface."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Lightweight health check for load balancers and Docker (no DB/OCI)."""
    return {"status": "ok"}

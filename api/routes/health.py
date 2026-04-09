"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Lightweight health check for load balancers and Docker (no DB/OCI)."""
    return {"status": "ok"}

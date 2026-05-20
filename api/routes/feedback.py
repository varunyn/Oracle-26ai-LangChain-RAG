"""Feedback endpoint for the runtime API surface."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest
from api.settings import get_settings
from src.rag_agent.utils.langfuse_tracing import get_langfuse_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])

try:
    from src.rag_agent.utils.rag_feedback import RagFeedback as RagFeedbackClass
except ImportError:
    RagFeedbackClass = None  # type: ignore[assignment,misc]


def record_langfuse_feedback_score(request: FeedbackRequest) -> None:
    trace_id = (request.trace_id or "").strip()
    if not trace_id:
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name="user-rating",
            value=request.feedback,
            data_type="NUMERIC",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse feedback score failed: %s", exc)


@router.post("/feedback")
async def post_feedback(request: FeedbackRequest) -> dict[str, str]:
    settings = get_settings()
    if not getattr(settings, "ENABLE_USER_FEEDBACK", False):
        raise HTTPException(status_code=403, detail="User feedback is disabled")
    if RagFeedbackClass is None:
        raise HTTPException(status_code=503, detail="Feedback service not available")
    try:
        rag_feedback = RagFeedbackClass()
        await asyncio.to_thread(
            rag_feedback.insert_feedback,
            request.question,
            request.answer,
            request.feedback,
        )
        record_langfuse_feedback_score(request)
        return {"status": "ok"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Feedback insert error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save feedback")

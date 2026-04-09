"""Feedback endpoint: POST /api/feedback."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest
from api.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])

try:
    from src.rag_agent.utils.rag_feedback import RagFeedback as RagFeedbackClass
except ImportError:
    RagFeedbackClass = None  # type: ignore[assignment,misc]


@router.post("/feedback")
async def post_feedback(request: FeedbackRequest):
    s = get_settings()
    if not getattr(s, "ENABLE_USER_FEEDBACK", False):
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
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Feedback insert error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save feedback")

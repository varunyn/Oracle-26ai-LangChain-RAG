"""Feedback endpoint for the runtime API surface."""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest
from api.settings import get_settings
from src.rag_agent.utils.langfuse_tracing import get_langfuse_client, safe_flush

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])
USER_RATING_SCORE_NAME = "user-rating"

try:
    from src.rag_agent.utils.rag_feedback import RagFeedback as RagFeedbackClass
except ImportError:
    RagFeedbackClass = None  # type: ignore[assignment,misc]


def _get_langfuse_score_config_id(client: Any, score_name: str) -> str | None:
    score_configs_client = getattr(getattr(client, "api", None), "score_configs", None)
    get_score_configs = getattr(score_configs_client, "get", None)
    if not callable(get_score_configs):
        return None
    try:
        response = get_score_configs(limit=100)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse score config lookup failed: %s", exc)
        return None
    data = getattr(response, "data", None)
    if not isinstance(data, list):
        return None
    for item in data:
        name = getattr(item, "name", None)
        is_archived = getattr(item, "is_archived", getattr(item, "isArchived", False))
        config_id = getattr(item, "id", None)
        if name == score_name and is_archived is not True and isinstance(config_id, str):
            return config_id
    return None


def record_langfuse_feedback_score(request: FeedbackRequest) -> None:
    trace_id = (request.trace_id or "").strip()
    if not trace_id:
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        score_kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": USER_RATING_SCORE_NAME,
            "value": request.feedback,
            "data_type": "NUMERIC",
        }
        config_id = _get_langfuse_score_config_id(client, USER_RATING_SCORE_NAME)
        if config_id:
            score_kwargs["config_id"] = config_id
        client.create_score(**score_kwargs)
        safe_flush()
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

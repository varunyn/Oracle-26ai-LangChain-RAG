"""Config endpoint: GET /api/config."""

from fastapi import APIRouter, Request

from api.deps.request import get_settings as get_settings_dep
from api.settings import Settings

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config(request: Request):
    settings: Settings = get_settings_dep(request)
    return {
        "region": settings.REGION,
        "embed_model_id": settings.EMBED_MODEL_ID,
        "model_list": settings.MODEL_LIST,
        "model_display_names": settings.MODEL_DISPLAY_NAMES,
        "collection_list": settings.COLLECTION_LIST or [settings.DEFAULT_COLLECTION],
        "enable_user_feedback": settings.ENABLE_USER_FEEDBACK,
    }

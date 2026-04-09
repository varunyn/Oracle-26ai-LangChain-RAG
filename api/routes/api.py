"""
API router aggregation module.

Composes all individual routers into a single router for inclusion in the main app.
"""

from fastapi import APIRouter

from . import chat, config_router, documents, feedback, graph, health, mcp, suggestions

router = APIRouter()

# Include routers in the same order as currently done in main.py
router.include_router(health.router)
router.include_router(graph.router)
router.include_router(chat.router)
router.include_router(mcp.router)
router.include_router(suggestions.router)
router.include_router(config_router.router)
router.include_router(feedback.router)
router.include_router(documents.router)

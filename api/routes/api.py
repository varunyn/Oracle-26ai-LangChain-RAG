"""
API router aggregation module.

Composes all individual routers into a single router for inclusion in the main app.
"""

from fastapi import APIRouter

from src.rag_agent.runtime import config, documents, feedback, health, langgraph_server, suggestions

router = APIRouter()

# Include routers in the same order as currently done in main.py
router.include_router(health.router)
router.include_router(langgraph_server.router)
router.include_router(suggestions.router)
router.include_router(config.router)
router.include_router(feedback.router)
router.include_router(documents.router)

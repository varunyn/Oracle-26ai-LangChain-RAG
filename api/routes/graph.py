"""Graph visualization endpoints. Exposes the RAG workflow as Mermaid (per LangGraph use-graph-api)."""

from typing import cast

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from api.dependencies import get_graph_service

router = APIRouter(tags=["graph"], prefix="/graph")


@router.get("/mermaid", response_class=PlainTextResponse)
def get_graph_mermaid(graph_service=Depends(get_graph_service)) -> str:
    """
    Return the RAG workflow as Mermaid diagram text.

    Uses LangGraph's get_graph().draw_mermaid() (see docs:
    https://docs.langchain.com/oss/python/langgraph/use-graph-api#mermaid).
    You can paste the response into Mermaid Live Editor or render in docs.
    """
    return cast(str, graph_service.get_graph().draw_mermaid())

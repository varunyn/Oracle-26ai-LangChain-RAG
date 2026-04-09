"""Legacy MCP chat endpoint: /api/mcp/chat.

This route remains for compatibility, but the active MCP-enabled chat path in this repo is
`/api/chat` with `mode="mcp"` or `mode="mixed"`.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.dependencies import get_mcp_servers_config_cached as get_mcp_servers_config
from api.schemas import McpChatRequest

logger = logging.getLogger(__name__)

MEDIA_TYPE_SSE = "text/event-stream"


def sse_chunk(delta: str, index: int = 0, finish_reason: str | None = None) -> str:
    obj = {
        "id": "",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": index,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(obj)}\n\n"


router = APIRouter(prefix="/api/mcp", tags=["mcp"])

AgentWithMCP: Any = None
default_jwt_supplier: Any = None
try:
    from src.rag_agent.infrastructure.llm_with_mcp import AgentWithMCP, default_jwt_supplier
except ImportError:
    pass


async def _mcp_chat_stream(request: McpChatRequest):
    """Run the legacy MCP chat path and stream the final answer as SSE.

    In the current repo state this may return an availability error because AgentWithMCP is a
    compatibility stub. The supported MCP path is `/api/chat`.
    """
    if not request.messages:
        yield f"data: {json.dumps({'error': 'Messages array is required'})}\n\n"
        return
    _cfg = get_mcp_servers_config()
    _entry = _cfg.get("default") or (next(iter(_cfg.values()), None) if _cfg else None)
    mcp_url = (request.mcp_url or "").strip() or ((_entry or {}).get("url") if _entry else None)
    if not mcp_url:
        yield f"data: {json.dumps({'error': 'No MCP URL (set MCP_SERVERS_CONFIG in .env or send mcp_url)'})}\n\n"
        return
    history = []
    for m in request.messages[:-1]:
        role = (m.role or "").lower()
        if role in ("user", "assistant") and (m.content or "").strip():
            history.append({"role": role, "content": (m.content or "").strip()})
    last = request.messages[-1]
    question = (last.content or "").strip() if last else ""
    if not question:
        yield f"data: {json.dumps({'error': 'Empty or missing last message'})}\n\n"
        return
    if AgentWithMCP is None or default_jwt_supplier is None:
        yield f"data: {json.dumps({'error': 'MCP integration not available'})}\n\n"
        return
    try:
        agent = await AgentWithMCP.create(
            mcp_url=mcp_url,
            jwt_supplier=default_jwt_supplier,
            timeout=60,
        )
        answer = await agent.answer(question, history=history or None)
    except Exception as e:
        logger.exception("MCP chat error", exc_info=e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
    yield sse_chunk(answer or "", index=0)
    yield sse_chunk("", index=0, finish_reason="stop")
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def api_mcp_chat(request: McpChatRequest):
    """Legacy MCP chat endpoint.

    This route remains in the API surface for compatibility, but it is not the primary supported
    MCP integration path. Use `/api/chat` with `mode="mcp"` or `mode="mixed"` for active MCP use.
    """
    if request.stream:
        return StreamingResponse(
            _mcp_chat_stream(request),
            media_type=MEDIA_TYPE_SSE,
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    content_parts = []
    async for chunk in _mcp_chat_stream(request):
        if chunk.strip().startswith("data: ") and chunk.strip() != "data: [DONE]\n\n":
            data_str = chunk.strip()[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
                if "error" in obj:
                    return {"error": obj["error"]}
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    content_parts.append(delta)
            except json.JSONDecodeError:
                pass
    return {"content": "".join(content_parts)}

"""
Legacy MCP compatibility surface for API/UI imports.

The active MCP path in this repo is `/api/chat` with `mode="mcp"` or `mode="mixed"`,
which flows through the LangGraph nodes and `mcp_agent.get_mcp_answer()`.
This module intentionally exports stubs so legacy imports continue to resolve without reviving
the old `/api/mcp/chat` implementation. When these stubs are in place, `/api/mcp/chat` reports
"MCP integration not available" by design.
"""

# Stubs so "from ... llm_with_mcp import AgentWithMCP, default_jwt_supplier" succeeds.
# Real MCP chat uses mcp_agent.get_mcp_answer() with get_llm() (ChatOCIGenAI), same as RAG.
AgentWithMCP = None
default_jwt_supplier = None

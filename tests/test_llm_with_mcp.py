"""
Test cases for src/rag_agent/infrastructure/llm_with_mcp.py

The module is a legacy stub: it exports AgentWithMCP and default_jwt_supplier as None
so api/rag_agent_api and ui/ui_mcp_agent can import without error. The real MCP path
uses mcp_agent.get_mcp_answer() and oci_models.get_llm() (ChatOCIGenAI).
"""

import pytest

try:
    from src.rag_agent.infrastructure.llm_with_mcp import AgentWithMCP, default_jwt_supplier
except ImportError as e:
    pytest.skip(f"llm_with_mcp not available: {e}", allow_module_level=True)


def test_llm_with_mcp_exports_stubs():
    """Module exports None stubs for API/UI compatibility."""
    assert AgentWithMCP is None
    assert default_jwt_supplier is None


def test_llm_with_mcp_module_docstring_marks_legacy_stub():
    """Module docs should clarify the supported MCP path and legacy stub status."""
    import src.rag_agent.infrastructure.llm_with_mcp as module

    assert module.__doc__ is not None
    assert "Legacy MCP compatibility surface" in module.__doc__
    assert "`/api/chat`" in module.__doc__
    assert "`/api/mcp/chat`" in module.__doc__

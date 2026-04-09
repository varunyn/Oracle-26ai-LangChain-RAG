"""
Tests for MCP agent prompt hashes to ensure prompt content is locked.
"""

import hashlib

from src.rag_agent.infrastructure import mcp_agent as mcp_agent_module
from src.rag_agent.prompts.mcp_agent_prompts import SYSTEM_PROMPT as PROMPTS_SYSTEM_PROMPT
from src.rag_agent.prompts.mcp_agent_prompts import (
    SYSTEM_PROMPT_MIXED as PROMPTS_SYSTEM_PROMPT_MIXED,
)


def test_system_prompt_hash_locked():
    """Ensure SYSTEM_PROMPT content is locked by SHA256 hash."""
    expected_hash = "10d20009cde7779aa823169ade4ea5a81b6f4a6536d9d82e9ef07e2850f95d71"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT hash changed: {actual_hash}"


def test_system_prompt_mixed_hash_locked():
    """Ensure SYSTEM_PROMPT_MIXED content is locked by SHA256 hash."""
    expected_hash = "5d48220995265a312d6b431649fea26e1e2a63c37e886fa4e9c1d66ef3e97440"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT_MIXED.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT_MIXED hash changed: {actual_hash}"


def test_prompts_consistency():
    """Ensure prompts imported in mcp_agent match the source prompts module."""
    assert mcp_agent_module.SYSTEM_PROMPT == PROMPTS_SYSTEM_PROMPT
    assert mcp_agent_module.SYSTEM_PROMPT_MIXED == PROMPTS_SYSTEM_PROMPT_MIXED

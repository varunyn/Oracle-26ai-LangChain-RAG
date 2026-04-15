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
    expected_hash = "f56f7b385990c5d3f03d4f8f32bc9df9e8b96de06e444592165d08bd4e78fca3"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT hash changed: {actual_hash}"


def test_system_prompt_mixed_hash_locked():
    """Ensure SYSTEM_PROMPT_MIXED content is locked by SHA256 hash."""
    expected_hash = "e603888971d7d2528cb87e6b985c9480bd95ad1b355045b02d4a62daa40676ed"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT_MIXED.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT_MIXED hash changed: {actual_hash}"


def test_prompts_consistency():
    """Ensure prompts imported in mcp_agent match the source prompts module."""
    assert mcp_agent_module.SYSTEM_PROMPT == PROMPTS_SYSTEM_PROMPT
    assert mcp_agent_module.SYSTEM_PROMPT_MIXED == PROMPTS_SYSTEM_PROMPT_MIXED

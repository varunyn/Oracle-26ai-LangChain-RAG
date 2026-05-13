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
    expected_hash = "557b2f688d28e5a33acecb1876d5705fcc084cf691080b9d247667bd2d8ac961"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT hash changed: {actual_hash}"


def test_system_prompt_mixed_hash_locked():
    """Ensure SYSTEM_PROMPT_MIXED content is locked by SHA256 hash."""
    expected_hash = "81cecece6a92d908735409fe017563bb5fe149c61d92e5b109f49a69cb781749"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT_MIXED.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT_MIXED hash changed: {actual_hash}"


def test_prompts_consistency():
    """Ensure prompts imported in mcp_agent match the source prompts module."""
    assert mcp_agent_module.SYSTEM_PROMPT == PROMPTS_SYSTEM_PROMPT
    assert mcp_agent_module.SYSTEM_PROMPT_MIXED == PROMPTS_SYSTEM_PROMPT_MIXED

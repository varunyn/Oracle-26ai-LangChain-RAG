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
    expected_hash = "1d9c12159b8f777a0da04ba8dea1ef24aa9df7b1ab0927fee7b0bb6d53696fdd"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT hash changed: {actual_hash}"


def test_system_prompt_mixed_hash_locked():
    """Ensure SYSTEM_PROMPT_MIXED content is locked by SHA256 hash."""
    expected_hash = "185259740c27f000f0c5d20f51d12c2b6bc6169846222ba972988f00f3ec552e"
    actual_hash = hashlib.sha256(mcp_agent_module.SYSTEM_PROMPT_MIXED.encode("utf-8")).hexdigest()
    assert actual_hash == expected_hash, f"SYSTEM_PROMPT_MIXED hash changed: {actual_hash}"


def test_prompts_consistency():
    """Ensure prompts imported in mcp_agent match the source prompts module."""
    assert mcp_agent_module.SYSTEM_PROMPT == PROMPTS_SYSTEM_PROMPT
    assert mcp_agent_module.SYSTEM_PROMPT_MIXED == PROMPTS_SYSTEM_PROMPT_MIXED


def test_mixed_prompt_disallows_general_knowledge_after_empty_oracle_retrieval():
    assert (
        "do not answer from general knowledge" in PROMPTS_SYSTEM_PROMPT_MIXED
    )
    assert (
        "selected Oracle collection does not contain the answer"
        in PROMPTS_SYSTEM_PROMPT_MIXED
    )

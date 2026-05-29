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
    expected_hash = "2b3d38a0ebcb942a017af9e62ff02ddab0a6c9c781deac2e36e4b7b6e05d625a"
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

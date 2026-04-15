"""Utility functions for the RAG agent."""

import json
import logging
import re
from typing import Protocol

from langchain_core.documents import Document


class MCPToolLike(Protocol):
    name: str
    description: str
    input_schema: object


def get_console_logger(name: str = "ConsoleLogger", level: str = "INFO") -> logging.Logger:
    """
    Return a logger for console (and OTLP when setup_logging() has been called).

    When setup_logging() is used, loggers propagate to root and are exported via
    OTLP to the configured collector (and optionally to console) with request_id.

    Best practice: In FastAPI/application code, prefer logging.getLogger(__name__)
    so logger names follow module hierarchy (e.g. api.main). Use this
    helper for scripts or code that may run without the API (e.g. populate scripts).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If root already has handlers (file logging configured), just propagate
    root = logging.getLogger()
    if root.handlers:
        logger.propagate = True
        return logger

    # Fallback: console-only (no setup_logging called)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def extract_text_triple_backticks(_text: str) -> str:
    """
    Extract the first text block enclosed between triple backticks (```) from a string.
    If none found, returns the entire text.
    """
    logger = logging.getLogger(__name__)
    pattern = r"```(.*?)```"
    try:
        blocks = [block.strip() for block in re.findall(pattern, _text, re.DOTALL)]
        return blocks[0] if blocks else _text
    except (IndexError, AttributeError) as e:
        logger.debug("extract_text_triple_backticks: no block found, using full text: %s", e)
        return _text


def extract_json_from_text(
    text: str | None, *, allow_markdown_block: bool = True
) -> dict[str, object] | None:
    """Extract the first JSON object from arbitrary LLM text.

    - Optionally strip ````json``` blocks before searching.
    - Returns ``None`` when no JSON could be parsed.
    """
    if not text or not isinstance(text, str):
        return None

    chunk = text.strip()
    if allow_markdown_block:
        block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", chunk)
        if block_match:
            chunk = block_match.group(1).strip()

    if not chunk.startswith("{"):
        json_match = re.search(r"\{.*\}", chunk, re.DOTALL)
        chunk = json_match.group(0).strip() if json_match else ""

    if not chunk.startswith("{"):
        return None

    try:
        parsed = json.loads(chunk)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def docs_serializable(docs: list[Document]) -> list[dict[str, object]]:
    """
    Convert LangChain documents to JSON-serializable dicts (page_content, metadata).
    Used by the streaming API.
    """
    return [{"page_content": doc.page_content, "metadata": doc.metadata or {}} for doc in docs]


def print_mcp_available_tools(tools: list[MCPToolLike]) -> None:
    """Print the available MCP tools in a readable format (name, description, input schema)."""
    print("\n--- MCP Available tools:")
    for tool in tools:
        print(f"Tool: {tool.name} - {tool.description}")
        print("Input Schema:")
        pretty_schema = json.dumps(tool.input_schema, indent=4, sort_keys=True)
        print(pretty_schema)
        print("")

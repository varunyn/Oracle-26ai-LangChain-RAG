"""Verify native LangChain tool calling with ChatOCIGenAI.

Run:
    uv run python scripts/verify_oci_native_tool_calling.py
    uv run python scripts/verify_oci_native_tool_calling.py --model-id xai.grok-4.20-0309-reasoning

The script binds small local tools to the configured OCI chat model and prints
whether the model returned structured ``AIMessage.tool_calls`` instead of plain
text that merely describes tool use.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from langchain_core.tools import tool

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.settings import get_settings
from src.rag_agent.infrastructure.oci_models import get_llm


@tool
def list_documents(folder_name: str) -> dict[str, object]:
    """List document file names in a business folder."""
    return {
        "folderName": folder_name,
        "documents": [
            {"fileName": "a.pdf", "filePath": folder_name},
            {"fileName": "b.pdf", "filePath": folder_name},
        ],
    }


@tool
def classify_document(file_name: str, file_path: str) -> dict[str, str]:
    """Classify a document by file name and path."""
    return {
        "fileName": file_name,
        "filePath": file_path,
        "classification": "INVOICE" if file_name.endswith(".pdf") else "OTHER",
    }


def _message_content_preview(message: object, *, max_len: int = 500) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                value = item.get("text") or item.get("content") or ""
                parts.append(str(value))
            else:
                parts.append(str(item))
        text = " ".join(part for part in parts if part).strip()
    else:
        text = str(content or "")
    return text[:max_len]


def _tool_call_name(tool_call: object) -> str:
    if isinstance(tool_call, Mapping):
        name = tool_call.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            function_name = function.get("name")
            if isinstance(function_name, str) and function_name.strip():
                return function_name.strip()
    return ""


def summarize_tool_call_response(message: object) -> dict[str, object]:
    """Return a compact, JSON-serializable native tool-call summary."""
    raw_tool_calls = getattr(message, "tool_calls", None)
    tool_calls = (
        list(raw_tool_calls)
        if isinstance(raw_tool_calls, Sequence) and not isinstance(raw_tool_calls, (str, bytes))
        else []
    )
    tool_names = [name for call in tool_calls if (name := _tool_call_name(call))]
    return {
        "native_tool_calls_detected": bool(tool_calls),
        "tool_call_count": len(tool_calls),
        "tool_names": tool_names,
        "content_preview": _message_content_preview(message),
    }


def run_probe(*, model_id: str | None, prompt: str) -> dict[str, object]:
    """Bind local tools to ChatOCIGenAI and invoke a native tool-call probe."""
    llm = get_llm(model_id=model_id)
    if not hasattr(llm, "bind_tools"):
        raise RuntimeError(f"{type(llm).__name__} does not expose bind_tools().")
    bound = llm.bind_tools([list_documents, classify_document])
    message = bound.invoke(prompt)
    summary = summarize_tool_call_response(message)
    summary["model_id"] = model_id or get_settings().LLM_MODEL_ID
    summary["message_type"] = type(message).__name__
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether ChatOCIGenAI returns structured LangChain tool calls."
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="OCI model id to test. Defaults to LLM_MODEL_ID from settings.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Use the available tools. First list documents in /invoices, then classify each "
            "returned document. Return tool calls only when tools are needed."
        ),
        help="Prompt to send to the tool-bound model.",
    )
    args = parser.parse_args()

    try:
        summary = run_probe(model_id=args.model_id, prompt=args.prompt)
    except Exception as exc:
        print(f"Native tool-calling probe failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary.get("native_tool_calls_detected"):
        print(
            "No structured AIMessage.tool_calls were detected. The model may be responding "
            "with plain text instead of native tool calls.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

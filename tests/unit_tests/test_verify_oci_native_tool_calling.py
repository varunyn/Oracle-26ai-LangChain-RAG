from __future__ import annotations

from types import SimpleNamespace

from scripts.verify_oci_native_tool_calling import summarize_tool_call_response


def test_summarize_tool_call_response_reports_structured_tool_calls() -> None:
    message = SimpleNamespace(
        content="",
        tool_calls=[
            {"name": "list_documents", "args": {"folder_name": "/invoices"}},
            {"name": "classify_document", "args": {"file_name": "a.pdf", "file_path": "/invoices"}},
        ],
    )

    summary = summarize_tool_call_response(message)

    assert summary["native_tool_calls_detected"] is True
    assert summary["tool_call_count"] == 2
    assert summary["tool_names"] == ["list_documents", "classify_document"]
    assert summary["content_preview"] == ""


def test_summarize_tool_call_response_reports_text_only_output() -> None:
    message = SimpleNamespace(content="I will call list_documents('/invoices')", tool_calls=[])

    summary = summarize_tool_call_response(message)

    assert summary["native_tool_calls_detected"] is False
    assert summary["tool_call_count"] == 0
    assert summary["tool_names"] == []
    assert summary["content_preview"] == "I will call list_documents('/invoices')"

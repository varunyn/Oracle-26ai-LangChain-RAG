"""Verification test: run 10 random queries from a fixed set (RAG + tool mix).

Used for checklist 6.4 in docs/ORACLE-LANGCHAIN-STACK.md. Each run picks randomly
from: equation solve (tool), OCI CLI config (RAG), integral (tool).

Ensures setup_logging() is called so logs go to OTLP/console when the test runs
(same as when the API server is up).

To see the results table, run: pytest tests/test_verification_queries.py -v -s
"""

import asyncio
import os
import random
import urllib.error
import urllib.request
from typing import Any, TypedDict
from urllib.parse import urlparse, urlunparse

import pytest

from api.settings import get_settings

# Import after path is set (conftest / project root); API adds project root to path

os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")

run_rag_and_get_answer: Any
ChatMessage: Any
try:
    from api.rag_agent_api import (
        ChatMessage as _ChatMessage,
    )
    from api.rag_agent_api import (
        run_rag_and_get_answer as _run_rag_and_get_answer,
    )

    ChatMessage = _ChatMessage
    run_rag_and_get_answer = _run_rag_and_get_answer
except ImportError:
    run_rag_and_get_answer = None
    ChatMessage = None

setup_logging: Any
set_request_id: Any
try:
    from src.rag_agent.utils.logging_config import (
        set_request_id as _set_request_id,
    )
    from src.rag_agent.utils.logging_config import (
        setup_logging as _setup_logging,
    )

    setup_logging = _setup_logging
    set_request_id = _set_request_id
except ImportError:
    setup_logging = None
    set_request_id = None

get_mcp_servers_config: Any
init_code_mode_client: Any
get_code_mode_client: Any
try:
    from src.rag_agent.infrastructure.code_mode_client import (
        get_code_mode_client as _get_code_mode_client,
    )
    from src.rag_agent.infrastructure.code_mode_client import (
        init_code_mode_client as _init_code_mode_client,
    )
    from src.rag_agent.infrastructure.mcp_settings import (
        get_mcp_servers_config as _get_mcp_servers_config,
    )

    get_mcp_servers_config = _get_mcp_servers_config
    init_code_mode_client = _init_code_mode_client
    get_code_mode_client = _get_code_mode_client
except ImportError:
    get_mcp_servers_config = None
    init_code_mode_client = None
    get_code_mode_client = None

# Fixed set of verification queries: tool-heavy, RAG-heavy, tool-heavy
VERIFICATION_QUERIES = [
    "Solve the following equation: x^2 - 5x + 6 = 0",
    "How do I integrate my visual application with a Git repository?",
    "Calculate the integral of x^2 * e^x.",
]

# Queries that should use MCP tools (calculator); RAG-only fallback "I don't know" is not acceptable
TOOL_QUERIES = {VERIFICATION_QUERIES[0], VERIFICATION_QUERIES[2]}

NUM_RANDOM_RUNS = 10


class ModeTestCase(TypedDict):
    mode: str
    question: str
    expect_mcp: bool


MODE_TEST_CASES: list[ModeTestCase] = [
    {"mode": "rag", "question": "how to config oci cli in linux", "expect_mcp": False},
    {
        "mode": "direct",
        "question": "Explain what the OCI CLI is in one sentence.",
        "expect_mcp": False,
    },
    {
        "mode": "mcp",
        "question": "Solve the following equation: x^2 - 5x + 6 = 0",
        "expect_mcp": True,
    },
    {
        "mode": "mixed",
        "question": "Calculate the integral of x^2 * e^x.",
        "expect_mcp": True,
    },
]

# Use test table for vector search so RAG queries (e.g. OCI CLI) hit test data
VERIFICATION_COLLECTION = "RAG_KNOWLEDGE_BASE_TEST"

# Table column widths for result summary (question and preview truncated)
_COL_RUN = 4
_COL_TYPE = 6
_COL_QUESTION = 48
_COL_MCP = 4
_COL_TOOLS = 20
_COL_LEN = 6
_COL_STATUS = 8
_COL_PREVIEW = 44


class VerificationRow(TypedDict):
    run: int
    type: str
    question: str
    mcp_used: bool
    mcp_tools_used: list[str]
    answer_len: int
    status: str
    answer_preview: str


def _is_tool_query(question: str) -> bool:
    return question in TOOL_QUERIES


def _trunc(s: str, width: int, suffix: str = "…") -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= width:
        return s
    return s[: max(0, width - len(suffix))] + suffix


def _cell(s: str, w: int) -> str:
    return _trunc(str(s), w).ljust(w)


def _build_row(
    *,
    run: int,
    row_type: str,
    question: str,
    mcp_used: bool,
    mcp_tools_used: list[str],
    answer_len: int,
    status: str,
    answer_preview: str,
) -> VerificationRow:
    return {
        "run": run,
        "type": row_type,
        "question": question,
        "mcp_used": mcp_used,
        "mcp_tools_used": mcp_tools_used,
        "answer_len": answer_len,
        "status": status,
        "answer_preview": answer_preview,
    }


def _check_mcp_available() -> bool:
    if get_mcp_servers_config is None:
        return False
    mcp_config = get_mcp_servers_config()
    if not mcp_config:
        return False
    for entry in mcp_config.values():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url:
            urls_to_try = [url]
            parsed = urlparse(url)
            if parsed.hostname == "host.docker.internal":
                urls_to_try.append(
                    urlunparse(parsed._replace(netloc=f"localhost:{parsed.port or 80}"))
                )
            for candidate in urls_to_try:
                try:
                    urllib.request.urlopen(candidate, timeout=5)
                    return True
                except urllib.error.HTTPError:
                    return True
                except Exception:
                    continue
    return False


_code_mode_client_ready: bool | None = None


def _ensure_code_mode_client() -> bool:
    global _code_mode_client_ready
    if _code_mode_client_ready is not None:
        return _code_mode_client_ready
    if not get_settings().CODE_MODE_ENABLED:
        _code_mode_client_ready = False
        return _code_mode_client_ready
    if init_code_mode_client is None or get_code_mode_client is None:
        _code_mode_client_ready = False
        return _code_mode_client_ready
    try:
        asyncio.run(init_code_mode_client())
        _ = get_code_mode_client()
        _code_mode_client_ready = True
        return _code_mode_client_ready
    except Exception:
        _code_mode_client_ready = False
        return _code_mode_client_ready


def _table_sep() -> str:
    return (
        "+"
        + "-" * _COL_RUN
        + "+"
        + "-" * _COL_TYPE
        + "+"
        + "-" * _COL_QUESTION
        + "+"
        + "-" * _COL_MCP
        + "+"
        + "-" * _COL_TOOLS
        + "+"
        + "-" * _COL_LEN
        + "+"
        + "-" * _COL_STATUS
        + "+"
        + "-" * _COL_PREVIEW
        + "+"
    )


def _table_header() -> str:
    return (
        "|"
        + _cell("Run", _COL_RUN)
        + "|"
        + _cell("Type", _COL_TYPE)
        + "|"
        + _cell("Question", _COL_QUESTION)
        + "|"
        + _cell("MCP", _COL_MCP)
        + "|"
        + _cell("Tools", _COL_TOOLS)
        + "|"
        + _cell("Len", _COL_LEN)
        + "|"
        + _cell("Status", _COL_STATUS)
        + "|"
        + _cell("Answer preview", _COL_PREVIEW)
        + "|"
    )


def _table_row(r: VerificationRow) -> str:
    tools_str = ",".join(r["mcp_tools_used"] or []) or "—"
    return (
        "|"
        + _cell(str(r["run"]), _COL_RUN)
        + "|"
        + _cell(r["type"], _COL_TYPE)
        + "|"
        + _cell(r["question"], _COL_QUESTION)
        + "|"
        + _cell("yes" if r["mcp_used"] else "no", _COL_MCP)
        + "|"
        + _cell(tools_str, _COL_TOOLS)
        + "|"
        + _cell(str(r["answer_len"]), _COL_LEN)
        + "|"
        + _cell(r["status"], _COL_STATUS)
        + "|"
        + _cell(r["answer_preview"], _COL_PREVIEW)
        + "|"
    )


def _print_table_start(total: int) -> None:
    """Print progress message and table header so rows can be streamed below."""
    print()
    print(f"Running {total} verification queries (table will fill as each completes)...")
    print()
    sep = _table_sep()
    print("Verification queries – results")
    print(sep)
    print(_table_header())
    print(sep)


def _print_table_row(r: VerificationRow) -> None:
    """Print a single result row (call after each query completes)."""
    print(_table_row(r))


def _print_table_end(rows: list[VerificationRow]) -> None:
    """Print table footer and summary."""
    print(_table_sep())
    passed = sum(1 for r in rows if r["status"] == "OK")
    ok_tool = sum(1 for r in rows if r["type"] == "tool" and r["status"] == "OK")
    ok_rag = sum(1 for r in rows if r["type"] == "RAG" and r["status"] == "OK")
    print(f"Summary: {passed}/{len(rows)} passed  (tool: {ok_tool}, RAG: {ok_rag})")
    print()


def _print_results_table(rows: list[VerificationRow]) -> None:
    """Print a fixed-width table of run results (batch, for callers that have all rows)."""
    _print_table_start(len(rows))
    for r in rows:
        _print_table_row(r)
    _print_table_end(rows)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration verification queries",
)
@pytest.mark.skipif(
    run_rag_and_get_answer is None or ChatMessage is None,
    reason="api.rag_agent_api not importable (run from project root)",
)
def test_10_random_verification_queries(capsys: pytest.CaptureFixture[str]):
    """Run 10 queries from VERIFICATION_QUERIES; assert no error, non-empty answer, and tool queries use MCP if available."""
    if setup_logging is not None:
        setup_logging()
    random.seed(42)
    errors = []
    tool_failures = []
    rows: list[VerificationRow] = []

    # Check if MCP is configured and reachable
    mcp_available = _check_mcp_available()
    if not mcp_available or not _ensure_code_mode_client():
        pytest.skip("MCP servers not reachable for tool verification queries")
    from api.deps import request as deps
    from api.services.graph_service import GraphService

    graph_service = getattr(deps, "_fallback_graph_service", None) or GraphService()

    with capsys.disabled():
        _print_table_start(NUM_RANDOM_RUNS)

    for i in range(NUM_RANDOM_RUNS):
        if set_request_id is not None:
            set_request_id(f"verify-{i + 1}")
        question = random.choice(VERIFICATION_QUERIES)
        with capsys.disabled():
            print(f"  Run {i + 1}/{NUM_RANDOM_RUNS}: {_trunc(question, 60)}...")
        messages = [ChatMessage(role="user", content=question)]
        answer, err, _standalone, _citations, _docs, _usage, mcp_used, mcp_tools_used = (
            run_rag_and_get_answer(
                messages,
                collection_name=VERIFICATION_COLLECTION,
                graph_service=graph_service,
            )
        )
        q_type = "tool" if _is_tool_query(question) else "RAG"
        answer_text = (answer or "").strip()
        mcp_used_flag = bool(mcp_used)
        mcp_tools_used_list = [str(tool) for tool in (mcp_tools_used or [])]
        if err:
            errors.append((i + 1, question, err))
            status = "ERROR"
        elif not answer_text:
            status = "EMPTY"
        elif (
            _is_tool_query(question)
            and mcp_available
            and (not mcp_used_flag or not mcp_tools_used_list)
        ):
            tool_failures.append((i + 1, question, mcp_used, mcp_tools_used, answer_text[:80]))
            status = "NO_MCP"
        elif (
            _is_tool_query(question)
            and mcp_available
            and "i don't know the answer" in answer_text.lower()
        ):
            tool_failures.append(
                (i + 1, question, mcp_used, mcp_tools_used, "answer was I don't know")
            )
            status = "RAG_FALLBACK"
        else:
            status = "OK"
        row = _build_row(
            run=i + 1,
            row_type=q_type,
            question=question,
            mcp_used=mcp_used_flag,
            mcp_tools_used=mcp_tools_used_list,
            answer_len=len(answer_text),
            status=status,
            answer_preview=_trunc(answer_text, _COL_PREVIEW),
        )
        rows.append(row)
        with capsys.disabled():
            _print_table_row(rows[-1])

    with capsys.disabled():
        _print_table_end(rows)
    assert not errors, f"Failures: {errors}"
    empty = [r["run"] for r in rows if r["status"] == "EMPTY"]
    assert not empty, f"Empty answer in run(s): {empty}"
    assert (
        not tool_failures
    ), f"Tool queries must use MCP and not return 'I don't know': {tool_failures}"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration verification queries",
)
@pytest.mark.skipif(
    run_rag_and_get_answer is None or ChatMessage is None,
    reason="api.rag_agent_api not importable (run from project root)",
)
def test_mode_specific_verification_queries(capsys: pytest.CaptureFixture[str]):
    if setup_logging is not None:
        setup_logging()

    mcp_available = _check_mcp_available()
    if any(case["expect_mcp"] for case in MODE_TEST_CASES) and (
        not mcp_available or not _ensure_code_mode_client()
    ):
        pytest.skip("MCP servers not reachable for mode verification queries")

    from api.deps import request as deps
    from api.services.graph_service import GraphService

    graph_service = getattr(deps, "_fallback_graph_service", None) or GraphService()

    rows: list[VerificationRow] = []
    with capsys.disabled():
        _print_table_start(len(MODE_TEST_CASES))

    for i, case in enumerate(MODE_TEST_CASES, start=1):
        if set_request_id is not None:
            set_request_id(f"verify-mode-{i}")
        question = case["question"]
        mode = case["mode"]
        messages = [ChatMessage(role="user", content=question)]
        answer, err, _standalone, _citations, _docs, _usage, mcp_used, mcp_tools_used = (
            run_rag_and_get_answer(
                messages,
                collection_name=VERIFICATION_COLLECTION,
                graph_service=graph_service,
                mode=mode,
            )
        )
        answer_text = (answer or "").strip()
        mcp_used_flag = bool(mcp_used)
        mcp_tools_used_list = [str(tool) for tool in (mcp_tools_used or [])]
        status = "OK"
        if err:
            status = "ERROR"
        elif not answer_text:
            status = "EMPTY"
        elif case["expect_mcp"] and (not mcp_used_flag or not mcp_tools_used_list):
            status = "NO_MCP"
        elif not case["expect_mcp"] and mcp_used_flag:
            status = "UNEXPECTED_MCP"

        row = _build_row(
            run=i,
            row_type=mode,
            question=question,
            mcp_used=mcp_used_flag,
            mcp_tools_used=mcp_tools_used_list,
            answer_len=len(answer_text),
            status=status,
            answer_preview=_trunc(answer_text, _COL_PREVIEW),
        )
        rows.append(row)
        with capsys.disabled():
            _print_table_row(rows[-1])

    with capsys.disabled():
        _print_table_end(rows)

    failures = [r for r in rows if r["status"] not in ("OK",)]
    assert not failures, f"Mode verification failures: {failures}"

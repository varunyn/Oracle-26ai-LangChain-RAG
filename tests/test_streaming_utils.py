"""
Streaming test utilities for pytest.

Provides fixtures and helpers for testing SSE (Server-Sent Events) responses
with httpx AsyncClient configured for FastAPI ASGI apps.
"""

from collections.abc import Iterator

import httpx
import pytest
from httpx import ASGITransport

from api.main import app


@pytest.fixture
async def async_client():
    """AsyncClient configured with ASGITransport pointing at FastAPI app."""
    # Use ASGITransport to test against the actual FastAPI app
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


def parse_sse_stream(bytes_iter: Iterator[bytes]) -> Iterator[str]:
    """
    Parse SSE stream from incremental bytes, reconstructing lines across chunk boundaries.

    Consumes bytes incrementally and yields data: payload strings without the "data: " prefix.
    Handles partial lines that span multiple byte chunks.

    Args:
        bytes_iter: Iterator yielding byte chunks from the SSE stream

    Yields:
        str: The payload portion of each "data: ..." line (without "data: " prefix)
    """
    buffer = b""

    for chunk in bytes_iter:
        buffer += chunk

        # Process complete lines (ending with \n)
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line: str = line_bytes.decode("utf-8").rstrip("\r")  # Handle CRLF

            if line.startswith("data: "):
                yield line[6:]  # Remove "data: " prefix


async def collect_sse_data(response: httpx.Response, max_chunks: int = 1000) -> list[str]:
    """
    Collect SSE data payloads until [DONE] or end of stream.

    Reads the response stream incrementally, parses SSE frames, and collects
    all "data:" payloads until "[DONE]" is encountered (exclusive - [DONE] not included).

    Args:
        response: httpx Response object with stream=True
        max_chunks: Maximum number of chunks to read to prevent infinite loops

    Returns:
        list[str]: List of data payloads (without "data: " prefix)
    """
    data_payloads: list[str] = []
    buffer = b""
    chunks_read = 0

    async for chunk in response.aiter_bytes():
        chunks_read += 1
        if chunks_read > max_chunks:
            raise RuntimeError(f"Too many chunks read ({max_chunks}), possible infinite loop")

        buffer += chunk

        # Process complete lines (ending with \n)
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line: str = line_bytes.decode("utf-8").rstrip("\r")  # Handle CRLF

            if line.startswith("data: "):
                payload = line[6:]  # Remove "data: " prefix
                if payload == "[DONE]":
                    return data_payloads
                data_payloads.append(payload)

    return data_payloads


def test_sse_parser_handles_partial_chunks():
    """Test that SSE parser correctly reconstructs lines split across byte chunks."""

    # Simulate SSE stream split mid-line
    # Original: "data: {\"type\":\"start\",\"messageId\":\"abc\"}\n\n"
    chunks = [
        b'data: {"type":"start","messageId":"abc"}\n',  # Complete line
        b"\n",  # Empty line
        b'data: {"type":"text-delta","id":"abc:text",',  # Partial line
        b'"delta":"Hello"}\n\n',  # Continuation + complete line
        b"data: [DONE]\n\n",  # Final chunk
    ]

    payloads = list(parse_sse_stream(iter(chunks)))

    # Should have extracted 3 data payloads
    assert len(payloads) == 3
    assert payloads[0] == '{"type":"start","messageId":"abc"}'
    assert payloads[1] == '{"type":"text-delta","id":"abc:text","delta":"Hello"}'
    assert payloads[2] == "[DONE]"


def test_collect_sse_data_handles_partial_chunks():
    """Test that collect_sse_data correctly handles data lines split across async chunks."""
    import asyncio
    from unittest.mock import Mock

    # Create a mock response that yields chunks with a data line split across boundaries
    mock_response = Mock()

    # Create an async generator for aiter_bytes
    async def mock_aiter_bytes():
        # Simulate: data: {"type":"start"}\n\n split across chunks
        yield b'data: {"type":"start"}\n'  # First part of data line
        yield b"\n"  # Empty line (end of SSE frame)
        yield b"data: [DONE]\n\n"  # Next frame

    mock_response.aiter_bytes = mock_aiter_bytes

    async def run_test():
        result = await collect_sse_data(mock_response)
        assert result == ['{"type":"start"}']

    asyncio.run(run_test())

from __future__ import annotations

import json
from typing import Any

import httpx


def command_envelope(payload: dict[str, object], *, command_id: int = 1) -> dict[str, object]:
    return {"id": command_id, "method": "run.start", "params": payload}


def _protocol_events_to_legacy_chunks(raw_stream: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    for line in raw_stream.splitlines():
        if not line.startswith(b"data: "):
            continue
        protocol_event = json.loads(line.removeprefix(b"data: "))
        if not isinstance(protocol_event, dict):
            continue
        method = protocol_event.get("method")
        if method not in {"values", "tools"}:
            continue
        params = protocol_event.get("params")
        if not isinstance(params, dict):
            continue
        data = params.get("data")
        chunks.append(f"event: {method}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
    return chunks


async def run_v1_command_stream(
    client: httpx.AsyncClient,
    *,
    thread_id: str,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    channels: list[str] | None = None,
) -> tuple[httpx.Response, list[bytes]]:
    from api.routes.langgraph_server import _protocol_events

    async with _protocol_events._condition:
        _protocol_events._events.pop(thread_id, None)

    command_response = await client.post(
        f"/api/langgraph/threads/{thread_id}/commands",
        headers=headers,
        json=command_envelope(payload),
    )
    if command_response.status_code != 200:
        return command_response, []

    async with client.stream(
        "POST",
        f"/api/langgraph/threads/{thread_id}/stream/events",
        headers=headers,
        json={"channels": channels or ["values", "tools", "lifecycle"], "depth": 1},
    ) as stream_response:
        stream_response.raise_for_status()
        raw_stream = await stream_response.aread()

    return command_response, _protocol_events_to_legacy_chunks(raw_stream)


def protocol_event_data(raw_stream: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_stream.splitlines():
        if line.startswith(b"data: "):
            event = json.loads(line.removeprefix(b"data: "))
            if isinstance(event, dict):
                events.append(event)
    return events

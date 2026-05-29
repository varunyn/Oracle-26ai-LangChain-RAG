from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine


def run_coroutine_sync(coro: Coroutine[object, object, object]) -> object:
    """Run a coroutine from sync code, including when an event loop is already active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("Thread runner did not return a value.")
    return result["value"]

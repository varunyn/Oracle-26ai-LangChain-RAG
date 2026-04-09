from __future__ import annotations

# pyright: reportAny=false, reportConstantRedefinition=false, reportExplicitAny=false
# pyright: reportUnreachable=false, reportUnusedCallResult=false, reportUnusedParameter=false
import asyncio
import logging
import threading
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from typing_extensions import override

from api.settings import get_settings
from src.rag_agent.core import config as core_config
from src.rag_agent.utils.context_window import estimate_tokens, messages_to_text

LangfuseRuntime: type[Any] | None
try:
    from langfuse import Langfuse as _LangfuseRuntime
except Exception:
    LangfuseRuntime = None  # type: ignore[assignment]
else:
    LangfuseRuntime = _LangfuseRuntime


logger = logging.getLogger(__name__)

DEFAULT_FLUSH_TIMEOUT = 0.2


_CLIENT_LOCK = threading.Lock()
_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_DISABLED = False
_DISABLE_REASON = ""


def set_langfuse_client(client: Any | None, *, disabled: bool = False) -> None:
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED, _DISABLE_REASON
    with _CLIENT_LOCK:
        _LANGFUSE_CLIENT = client
        _LANGFUSE_DISABLED = disabled
        _DISABLE_REASON = "overridden" if disabled else ""


def langfuse_enabled() -> bool:
    if _LANGFUSE_DISABLED:
        return False
    if LangfuseRuntime is None:
        return False
    return bool(core_config.ENABLE_LANGFUSE_TRACING)


def get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED, _DISABLE_REASON

    if not langfuse_enabled():
        return None

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    if LangfuseRuntime is None:
        _disable("langfuse package not installed")
        return None

    host = (getattr(get_settings(), "LANGFUSE_HOST", "") or "").strip()
    public_key = (getattr(get_settings(), "LANGFUSE_PUBLIC_KEY", "") or "").strip()
    secret_key = (getattr(get_settings(), "LANGFUSE_SECRET_KEY", "") or "").strip()

    if not (host and public_key and secret_key):
        _disable("missing LANGFUSE config")
        return None

    extra_kwargs: dict[str, Any] = {}
    environment = getattr(get_settings(), "LANGFUSE_TRACING_ENVIRONMENT", None) or getattr(
        get_settings(), "LANGFUSE_ENVIRONMENT", None
    )
    if environment:
        extra_kwargs["environment"] = environment
    release = getattr(get_settings(), "LANGFUSE_RELEASE", None)
    if release:
        extra_kwargs["release"] = release

    with _CLIENT_LOCK:
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT
        try:
            _LANGFUSE_CLIENT = LangfuseRuntime(
                public_key=public_key,
                secret_key=secret_key,
                base_url=host,
                **extra_kwargs,
            )
            logger.info("Langfuse client initialized (host=%s)", host)
        except Exception as exc:
            _disable(f"init failed: {exc}")
            return None
    return _LANGFUSE_CLIENT


def add_langfuse_callbacks(
    run_config: dict[str, Any],
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Add Langfuse CallbackHandler to run_config when Langfuse is enabled.

    Mutates run_config in place: adds callbacks and metadata so the LangGraph
    invoke/astream is fully traced (nodes, LLM, tools) in Langfuse. The handler
    reads langfuse_session_id and langfuse_user_id from config metadata.
    """
    if not langfuse_enabled():
        return
    if get_langfuse_client() is None:
        return
    try:
        from langfuse.langchain import CallbackHandler
    except Exception as exc:
        logger.debug("Langfuse LangChain callback not available: %s", exc)
        return
    callbacks = list(run_config.get("callbacks") or [])
    handler = CallbackHandler()
    callbacks.append(_TokenUsageCallback(handler))
    callbacks.append(handler)
    run_config["callbacks"] = callbacks
    metadata = dict(run_config.get("metadata") or {})
    model_id = (
        run_config.get("configurable", {}).get("model_id")
        if isinstance(run_config.get("configurable"), dict)
        else None
    )
    if isinstance(model_id, str) and model_id:
        metadata["ls_model_name"] = model_id
    metadata["langfuse_session_id"] = session_id or ""
    metadata["langfuse_user_id"] = user_id or ""
    run_config["metadata"] = metadata


class _TokenUsageCallback(BaseCallbackHandler):
    _handler: object
    _input_tokens_by_run: dict[UUID, int]

    def __init__(self, handler: object) -> None:
        super().__init__()
        self._handler = handler
        self._input_tokens_by_run = {}

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, object],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        try:
            if messages:
                flattened = [m for group in messages for m in group]
                prompt_text = messages_to_text(cast(list[object], flattened))
                tokens = estimate_tokens(prompt_text, _extract_model_id(dict(kwargs)))
                if tokens > 0:
                    self._input_tokens_by_run[run_id] = tokens
        except Exception as exc:
            logger.debug("Langfuse chat start token estimate failed: %s", exc)

    @override
    def on_llm_start(
        self,
        serialized: dict[str, object],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        try:
            if prompts:
                prompt_text = "\n".join(prompts)
                tokens = estimate_tokens(prompt_text, _extract_model_id(dict(kwargs)))
                if tokens > 0:
                    self._input_tokens_by_run[run_id] = tokens
        except Exception as exc:
            logger.debug("Langfuse llm start token estimate failed: %s", exc)

    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        try:
            input_tokens = self._input_tokens_by_run.pop(run_id, 0)
            usage, estimated = _merge_usage(response, dict(kwargs), input_tokens)
            if usage is None:
                return
            _inject_usage(response, usage)
            if estimated:
                _tag_estimated_usage(self._handler, run_id)
        except Exception as exc:
            logger.debug("Langfuse usage estimate failed: %s", exc)


def _merge_usage(
    response: LLMResult, kwargs: dict[str, object], input_override: int
) -> tuple[dict[str, int] | None, bool]:
    existing = _extract_usage_from_response(response)
    model_id = _extract_model_id(kwargs)
    prompt_tokens = input_override or _estimate_prompt_tokens(kwargs, model_id)
    completion_tokens = _estimate_completion_tokens(response, model_id)
    if existing is None:
        if prompt_tokens == 0 and completion_tokens == 0:
            return None, False
        return (
            {
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            True,
        )
    merged = dict(existing)
    estimated = False
    if merged.get("input", 0) == 0 and prompt_tokens > 0:
        merged["input"] = prompt_tokens
        estimated = True
    if merged.get("output", 0) == 0 and completion_tokens > 0:
        merged["output"] = completion_tokens
        estimated = True
    if merged.get("total", 0) == 0:
        merged["total"] = merged.get("input", 0) + merged.get("output", 0)
    return merged, estimated


def _inject_usage(response: LLMResult, usage: dict[str, int]) -> None:
    llm_output = getattr(response, "llm_output", None)
    if llm_output is None:
        response.llm_output = {"token_usage": usage}
    elif isinstance(llm_output, dict):
        llm_output_map = cast(dict[str, object], llm_output)
        if not llm_output_map.get("token_usage"):
            llm_output_map["token_usage"] = usage
    for generation in response.generations:
        for chunk in generation:
            gen_info = getattr(chunk, "generation_info", None)
            if gen_info is None:
                setattr(chunk, "generation_info", {"usage_metadata": usage})
                continue
            if isinstance(gen_info, dict):
                gen_info_map = cast(dict[str, object], gen_info)
                if not gen_info_map.get("usage_metadata"):
                    gen_info_map["usage_metadata"] = usage


def _tag_estimated_usage(handler: object, run_id: UUID) -> None:
    runs = getattr(handler, "runs", None)
    if not isinstance(runs, dict):
        return
    generation = cast(dict[UUID, object], runs).get(run_id)
    if generation is None:
        return
    update = getattr(generation, "update", None)
    if not callable(update):
        return
    try:
        update(tags=["usage_estimated=true"])
    except Exception:
        return


def _extract_usage_from_response(response: LLMResult) -> dict[str, int] | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        llm_output_map = cast(dict[str, object], llm_output)
        usage = llm_output_map.get("usage") or llm_output_map.get("token_usage")
        parsed = _normalize_usage(usage)
        if parsed:
            return parsed
    generations = response.generations
    for generation in generations:
        for chunk in generation:
            gen_info = getattr(chunk, "generation_info", None)
            if isinstance(gen_info, dict):
                gen_info_map = cast(dict[str, object], gen_info)
                parsed = _normalize_usage(gen_info_map.get("usage_metadata"))
                if parsed:
                    return parsed
            message = getattr(chunk, "message", None)
            if not isinstance(message, BaseMessage):
                continue
            parsed = _normalize_usage(getattr(message, "usage_metadata", None))
            if parsed:
                return parsed
    return None


def _normalize_usage(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage = cast(dict[str, object], raw)
    if {"input", "output", "total"}.issubset(usage.keys()):
        return _coerce_usage(usage)
    if {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(usage.keys()):
        return _coerce_usage(
            {
                "input": usage.get("prompt_tokens"),
                "output": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            }
        )
    if {"input_tokens", "output_tokens", "total_tokens"}.issubset(usage.keys()):
        return _coerce_usage(
            {
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "total": usage.get("total_tokens"),
            }
        )
    return None


def _coerce_usage(raw: dict[str, object]) -> dict[str, int] | None:
    def _to_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    input_tokens = _to_int(raw.get("input"))
    output_tokens = _to_int(raw.get("output"))
    total_tokens = _to_int(raw.get("total"))
    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }


def _extract_model_id(kwargs: dict[str, object]) -> str | None:
    invocation = kwargs.get("invocation_params")
    if isinstance(invocation, dict):
        invocation_map = cast(dict[str, object], invocation)
        model = invocation_map.get("model_id") or invocation_map.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _estimate_prompt_tokens(kwargs: dict[str, object], model_id: str | None) -> int:
    inputs = kwargs.get("inputs")
    if isinstance(inputs, dict):
        inputs_map = cast(dict[str, object], inputs)
        messages = inputs_map.get("messages")
        if isinstance(messages, list):
            return estimate_tokens(messages_to_text(cast(list[object], messages)), model_id)
        prompt = inputs_map.get("prompt") or inputs_map.get("input")
        if isinstance(prompt, str):
            return estimate_tokens(prompt, model_id)
    if isinstance(inputs, list):
        return estimate_tokens(messages_to_text(cast(list[object], inputs)), model_id)
    if isinstance(inputs, str):
        return estimate_tokens(inputs, model_id)
    return 0


def _estimate_completion_tokens(response: LLMResult, model_id: str | None) -> int:
    generations = response.generations
    if not generations or not generations[-1]:
        return 0
    chunk = generations[-1][-1]
    message = getattr(chunk, "message", None)
    if isinstance(message, BaseMessage):
        content = cast(object, message.content)
        if isinstance(content, str):
            return estimate_tokens(content, model_id)
    if isinstance(chunk, ChatGeneration):
        return estimate_tokens(chunk.text, model_id)
    return 0


def safe_flush(timeout: float = DEFAULT_FLUSH_TIMEOUT) -> None:
    client = get_langfuse_client()
    if client is None:
        return

    def _flush() -> None:
        try:
            client.flush()
        except Exception as exc:
            _disable(f"flush failed: {exc}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _flush()
        return

    async def _async_flush() -> None:
        await asyncio.to_thread(_flush)

    loop.create_task(_async_flush())


def _disable(reason: str) -> None:
    global _LANGFUSE_DISABLED, _DISABLE_REASON, _LANGFUSE_CLIENT
    if _LANGFUSE_DISABLED and reason == _DISABLE_REASON:
        return
    _LANGFUSE_DISABLED = True
    _DISABLE_REASON = reason
    _LANGFUSE_CLIENT = None
    logger.warning("Langfuse instrumentation disabled: %s", reason)

from __future__ import annotations

# pyright: reportAny=false, reportConstantRedefinition=false, reportExplicitAny=false
# pyright: reportUnreachable=false, reportUnusedCallResult=false, reportUnusedParameter=false
import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from typing_extensions import override

from api.settings import get_settings
from src.rag_agent.core.usage import normalize_usage
from src.rag_agent.runtime.observability import estimate_cost_usd
from src.rag_agent.utils.context_window import estimate_tokens, messages_to_text
from src.rag_agent.utils.logging_config import get_request_id

LangfuseRuntime: type[Any] | None
LangfusePropagateAttributes: Any | None
try:
    from langfuse import Langfuse as _LangfuseRuntime
    from langfuse import propagate_attributes as _propagate_attributes
except Exception:
    LangfuseRuntime = None  # type: ignore[assignment]
    LangfusePropagateAttributes = None
else:
    LangfuseRuntime = _LangfuseRuntime
    LangfusePropagateAttributes = _propagate_attributes


logger = logging.getLogger(__name__)

DEFAULT_FLUSH_TIMEOUT = 0.2
LANGFUSE_METADATA_VALUE_MAX_LENGTH = 200


_CLIENT_LOCK = threading.Lock()
_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_DISABLED = False
_DISABLE_REASON = ""


def _clean_optional_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null", "-"}:
        return None
    return normalized


def _env_or_settings_str(key: str) -> str:
    env_value = os.environ.get(key)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()
    value = getattr(get_settings(), key, "") or ""
    return value.strip() if isinstance(value, str) else ""


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _resolve_langfuse_host() -> str:
    host = _env_or_settings_str("LANGFUSE_HOST")
    if host and not _running_in_docker():
        return host
    if host in {"http://localhost:3300", "http://127.0.0.1:3300"} and _running_in_docker():
        rewritten = "http://langfuse-web:3000"
        logger.warning(
            "Rewriting LANGFUSE_HOST=%s to %s inside Docker so Langfuse OTEL export does not loop back to the container",
            host,
            rewritten,
        )
        return rewritten
    return host


def _configure_langfuse_sdk_environment() -> None:
    """Expose Pydantic-loaded v4 SDK settings through the SDK's env contract."""
    settings = get_settings()
    values = {
        "LANGFUSE_TRACING_ENVIRONMENT": getattr(
            settings, "LANGFUSE_TRACING_ENVIRONMENT", None
        ),
        "LANGFUSE_RELEASE": getattr(settings, "LANGFUSE_RELEASE", None),
    }
    for key, value in values.items():
        normalized = _clean_optional_identifier(value)
        if normalized and not os.environ.get(key):
            os.environ[key] = normalized


def _langfuse_metadata(metadata: dict[str, object]) -> dict[str, str]:
    """Return v4-compatible metadata without truncating trace input or output."""
    return {
        str(key): str(value)[:LANGFUSE_METADATA_VALUE_MAX_LENGTH]
        for key, value in metadata.items()
        if value is not None and str(value)
    }


@dataclass
class LangfuseChatTrace:
    trace_id: str | None = None
    parent_span_id: str | None = None
    trace_name: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    release: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] | None = None
    input_payload: object | None = None
    propagate_trace_attributes: bool = True
    _manager: Any | None = None
    _observation: Any | None = None
    _propagation_manager: Any | None = None

    @property
    def trace_context(self) -> dict[str, str] | None:
        if not self.trace_id:
            return None
        context = {"trace_id": self.trace_id}
        if self.parent_span_id:
            context["parent_span_id"] = self.parent_span_id
        return context

    def update_output(self, output: object) -> None:
        if self._observation is None:
            return
        update = getattr(self._observation, "update", None)
        try:
            if callable(update):
                update(output=output)
        except Exception as exc:
            logger.debug("Langfuse root trace output update failed: %s", exc)

    def update(self, **kwargs: object) -> None:
        if self._observation is None:
            return
        update = getattr(self._observation, "update", None)
        if not callable(update):
            return
        try:
            metadata = kwargs.get("metadata")
            if isinstance(metadata, dict):
                kwargs["metadata"] = _langfuse_metadata(cast(dict[str, object], metadata))
            update(**kwargs)
        except Exception as exc:
            logger.debug("Langfuse observation update failed: %s", exc)

    def update_outcome(self, outcome: str, **metadata: object) -> None:
        if self._observation is None:
            return
        payload = {"outcome": outcome, **metadata}
        level = "WARNING" if outcome == "truncated" else ("ERROR" if outcome == "error" else "DEFAULT")
        self.update(level=level, metadata=_langfuse_metadata(payload))

    def update_metadata(self, metadata: dict[str, str]) -> None:
        if self._observation is None:
            return
        update = getattr(self._observation, "update", None)
        try:
            if callable(update):
                update(metadata=_langfuse_metadata(cast(dict[str, object], metadata)))
        except Exception as exc:
            logger.debug("Langfuse root trace metadata update failed: %s", exc)

    def update_error(self, exc: BaseException) -> None:
        if self._observation is None:
            return
        update = getattr(self._observation, "update", None)
        if not callable(update):
            return
        try:
            update(level="ERROR", status_message=str(exc))
        except Exception as update_exc:
            logger.debug("Langfuse root trace error update failed: %s", update_exc)

    def __enter__(self) -> LangfuseChatTrace:
        if self._manager is None:
            return self
        try:
            self._observation = self._manager.__enter__()
            trace_id = getattr(self._observation, "trace_id", None)
            observation_id = getattr(self._observation, "id", None)
            self.trace_id = trace_id if isinstance(trace_id, str) and trace_id else self.trace_id
            self.parent_span_id = (
                observation_id
                if isinstance(observation_id, str) and observation_id
                else self.parent_span_id
            )
            if self.propagate_trace_attributes and LangfusePropagateAttributes is not None and (
                self.trace_name
                or self.session_id
                or self.user_id
                or self.metadata
                or self.tags
            ):
                propagation_kwargs: dict[str, Any] = {
                    "trace_name": self.trace_name,
                    "session_id": self.session_id,
                    "metadata": self.metadata or None,
                    "tags": self.tags or None,
                }
                if self.user_id:
                    propagation_kwargs["user_id"] = self.user_id
                self._propagation_manager = LangfusePropagateAttributes(**propagation_kwargs)
                self._propagation_manager.__enter__()
        except Exception as exc:
            _disable(f"root trace start failed: {exc}")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None and isinstance(exc, BaseException):
            self.update_error(exc)
        if self._propagation_manager is not None:
            try:
                self._propagation_manager.__exit__(exc_type, exc, traceback)
            except Exception as exit_exc:
                logger.debug("Langfuse trace attribute propagation close failed: %s", exit_exc)
        if self._manager is None:
            return
        try:
            self._manager.__exit__(exc_type, exc, traceback)
        except Exception as exit_exc:
            logger.debug("Langfuse root trace close failed: %s", exit_exc)


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
    return bool(getattr(get_settings(), "ENABLE_LANGFUSE_TRACING", False))


def get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED, _DISABLE_REASON

    if not langfuse_enabled():
        return None

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    if LangfuseRuntime is None:
        _disable("langfuse package not installed")
        return None

    host = _resolve_langfuse_host()
    public_key = _env_or_settings_str("LANGFUSE_PUBLIC_KEY")
    secret_key = _env_or_settings_str("LANGFUSE_SECRET_KEY")

    if not (host and public_key and secret_key):
        _disable("missing LANGFUSE config")
        return None

    _configure_langfuse_sdk_environment()
    extra_kwargs: dict[str, Any] = {}
    sample_rate = getattr(get_settings(), "LANGFUSE_SAMPLE_RATE", None)
    if isinstance(sample_rate, (int, float)) and not isinstance(sample_rate, bool):
        extra_kwargs["sample_rate"] = float(sample_rate)
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


def start_langfuse_chat_trace(
    *,
    enabled: bool | None,
    mode: str | None,
    model_id: str | None,
    session_id: str | None,
    thread_id: str | None,
    request_id: str | None = None,
    user_id: str | None = None,
    release: str | None = None,
    input_payload: object | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> LangfuseChatTrace:
    if enabled is not True:
        return LangfuseChatTrace()
    client = get_langfuse_client()
    if client is None:
        return LangfuseChatTrace()
    resolved_trace_name = trace_name or "chat.request"
    resolved_session_id = _clean_optional_identifier(session_id or thread_id)
    resolved_request_id = _clean_optional_identifier(request_id or get_request_id())
    resolved_release = _clean_optional_identifier(
        release or getattr(get_settings(), "LANGFUSE_RELEASE", None)
    )
    resolved_tags = tags or [
        tag
        for tag in (
            "chat",
            f"mode:{mode}" if mode else None,
            f"model:{model_id}" if model_id else None,
        )
        if tag is not None
    ]
    trace_metadata = _langfuse_metadata({
        key: value
        for key, value in {
            "mode": mode,
            "model_id": model_id,
            "thread_id": thread_id,
            "request_id": resolved_request_id,
            "session_id": resolved_session_id,
            "user_id": user_id,
            "release": resolved_release,
        }.items()
        if isinstance(value, str) and value
    })
    if metadata:
        trace_metadata.update(_langfuse_metadata(metadata))
    try:
        manager = client.start_as_current_observation(
            name=resolved_trace_name,
            as_type="chain",
            input=input_payload,
            metadata=trace_metadata or None,
        )
    except Exception as exc:
        _disable(f"root trace setup failed: {exc}")
        return LangfuseChatTrace()
    return LangfuseChatTrace(
        trace_name=resolved_trace_name,
        session_id=resolved_session_id,
        request_id=resolved_request_id,
        user_id=user_id,
        release=resolved_release,
        metadata=trace_metadata or None,
        tags=resolved_tags,
        input_payload=input_payload,
        _manager=manager,
    )


def add_langfuse_callbacks(
    run_config: dict[str, Any],
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    release: str | None = None,
    trace_context: dict[str, str] | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Add Langfuse CallbackHandler to run_config when Langfuse is enabled.

    Mutates run_config in place: adds callbacks and metadata so chat runtime
    invoke/stream execution is fully traced (LLM + tools) in Langfuse. The handler
    reads langfuse_session_id and langfuse_user_id from config metadata.
    """
    if not langfuse_enabled():
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        from langfuse.langchain import CallbackHandler
    except Exception as exc:
        logger.debug("Langfuse LangChain callback not available: %s", exc)
        return
    public_key = getattr(client, "public_key", None)
    if not isinstance(public_key, str) or not public_key.strip():
        public_key = (
            getattr(get_settings(), "LANGFUSE_PUBLIC_KEY", None) or None
        )
    handler_kwargs: dict[str, Any] = {"trace_context": cast(Any, trace_context)}
    if isinstance(public_key, str) and public_key.strip():
        handler_kwargs["public_key"] = public_key.strip()
    handler = CallbackHandler(**handler_kwargs)
    _append_callbacks(run_config, [_TokenUsageCallback(handler), handler])
    metadata = dict(run_config.get("metadata") or {})
    model_id = (
        run_config.get("configurable", {}).get("model_id")
        if isinstance(run_config.get("configurable"), dict)
        else None
    )
    if isinstance(model_id, str) and model_id:
        metadata["ls_model_name"] = model_id
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if request_id:
        metadata["request_id"] = request_id
    if release:
        metadata["release"] = release
    if trace_name:
        metadata["langfuse_trace_name"] = trace_name
    if tags:
        metadata["langfuse_tags"] = tags
    run_config["metadata"] = metadata


def _append_callbacks(run_config: dict[str, Any], callbacks: list[object]) -> None:
    existing = run_config.get("callbacks")
    if existing is None:
        run_config["callbacks"] = callbacks
        return
    add_handler = getattr(existing, "add_handler", None)
    if callable(add_handler):
        copy = getattr(existing, "copy", None)
        manager = copy() if callable(copy) else existing
        for callback in callbacks:
            manager.add_handler(cast(Any, callback))
        run_config["callbacks"] = manager
        return
    if isinstance(existing, list):
        run_config["callbacks"] = [*existing, *callbacks]
        return
    if isinstance(existing, tuple):
        run_config["callbacks"] = [*existing, *callbacks]
        return
    run_config["callbacks"] = [existing, *callbacks]


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
            _sanitize_response_usage_metadata_for_langfuse(response)
            input_tokens = self._input_tokens_by_run.pop(run_id, 0)
            usage, estimated = _merge_usage(response, dict(kwargs), input_tokens)
            if usage is None:
                return
            _inject_usage(response, usage)
            _update_generation_usage(self._handler, run_id, usage, _extract_model_id(dict(kwargs)))
            if estimated:
                _tag_estimated_usage(self._handler, run_id)
        except Exception as exc:
            logger.debug("Langfuse usage estimate failed: %s", exc)


def _sanitize_response_usage_metadata_for_langfuse(response: LLMResult) -> None:
    for generation in response.generations:
        for chunk in generation:
            gen_info = getattr(chunk, "generation_info", None)
            if isinstance(gen_info, dict) and isinstance(gen_info.get("usage_metadata"), dict):
                gen_info["usage_metadata"] = _sanitize_usage_metadata_for_langfuse(
                    cast(dict[str, object], gen_info["usage_metadata"])
                )

            message = getattr(chunk, "message", None)
            if not isinstance(message, BaseMessage):
                continue

            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                setattr(
                    message,
                    "usage_metadata",
                    _sanitize_usage_metadata_for_langfuse(cast(dict[str, object], usage_metadata)),
                )

            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                usage = response_metadata.get("usage")
                if isinstance(usage, dict):
                    response_metadata["usage"] = _sanitize_usage_metadata_for_langfuse(
                        cast(dict[str, object], usage)
                    )


def _sanitize_usage_metadata_for_langfuse(raw: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if key.endswith("_token_details") and isinstance(value, dict):
            nested = {
                str(nested_key): nested_value
                for nested_key, nested_value in cast(dict[object, object], value).items()
                if _is_langfuse_token_count(nested_value)
            }
            if nested:
                sanitized[key] = nested
            continue
        if key.endswith("_tokens_details") and isinstance(value, list):
            nested_items = [
                item
                for item in value
                if isinstance(item, dict) and _is_langfuse_token_count(item.get("token_count"))
            ]
            if nested_items:
                sanitized[key] = nested_items
            continue
        sanitized[key] = value
    return sanitized


def _is_langfuse_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _update_generation_usage(
    handler: object,
    run_id: UUID,
    usage: dict[str, int],
    model_id: str | None,
) -> None:
    runs = getattr(handler, "runs", None)
    if not isinstance(runs, dict):
        return
    generation = cast(dict[UUID, object], runs).get(run_id)
    if generation is None:
        return
    update = getattr(generation, "update", None)
    if not callable(update):
        return
    cost_usd, pricing_basis = estimate_cost_usd(model_id, usage)
    kwargs: dict[str, object] = {
        "usage_details": usage,
        "metadata": {"pricing_basis": pricing_basis},
    }
    if cost_usd is not None:
        kwargs["cost_details"] = {"total": cost_usd}
    try:
        update(**kwargs)
    except Exception:
        return


def _extract_usage_from_response(response: LLMResult) -> dict[str, int] | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        llm_output_map = cast(dict[str, object], llm_output)
        usage = llm_output_map.get("usage") or llm_output_map.get("token_usage")
        parsed = normalize_usage(usage)
        if parsed:
            return parsed
    generations = response.generations
    for generation in generations:
        for chunk in generation:
            gen_info = getattr(chunk, "generation_info", None)
            if isinstance(gen_info, dict):
                gen_info_map = cast(dict[str, object], gen_info)
                parsed = normalize_usage(gen_info_map.get("usage_metadata"))
                if parsed:
                    return parsed
            message = getattr(chunk, "message", None)
            if not isinstance(message, BaseMessage):
                continue
            parsed = normalize_usage(getattr(message, "usage_metadata", None))
            if parsed:
                return parsed
    return None


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


def safe_shutdown() -> None:
    """Best-effort Langfuse shutdown for long-running app teardown."""
    global _LANGFUSE_CLIENT
    client = get_langfuse_client()
    if client is None:
        return
    try:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            client.flush()
    except Exception as exc:
        logger.debug("Langfuse shutdown failed: %s", exc)
    finally:
        with _CLIENT_LOCK:
            _LANGFUSE_CLIENT = None


def _disable(reason: str) -> None:
    global _LANGFUSE_DISABLED, _DISABLE_REASON, _LANGFUSE_CLIENT
    if _LANGFUSE_DISABLED and reason == _DISABLE_REASON:
        return
    _LANGFUSE_DISABLED = True
    _DISABLE_REASON = reason
    _LANGFUSE_CLIENT = None
    logger.warning("Langfuse instrumentation disabled: %s", reason)

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import override

from api.settings import get_settings

from ...agent_state import FollowUpInterpretation, State
from ...core.node_logging import log_node_end, log_node_start
from ...infrastructure.oci_models import get_llm
from ...utils.context_window import calculate_context_usage, log_context_usage, messages_to_text

logger = logging.getLogger(__name__)

FOLLOWUP_INTERPRETER_PROMPT_TEMPLATE = """
You are a conversation-aware RAG controller.

Decide how the latest user request should be handled.

Return ONLY a JSON object with this exact schema:
{{
  "intent": "retrieve" | "reformat",
  "standalone_question": string | null,
  "response_instruction": string | null,
  "reasoning": string
}}

Rules:
- Use intent="retrieve" when the user is asking for new factual content, clarification, or follow-up information that should trigger retrieval.
- For intent="retrieve", rewrite the latest user request into a standalone retrieval-ready question using the chat history when necessary.
- Use intent="reformat" when the user is asking to transform, restyle, retry, summarize, bulletize, shorten, translate, or otherwise re-present the most recent grounded answer without needing new retrieval.
- For intent="reformat", set standalone_question to null and describe the presentation change in response_instruction.
- Do not hardcode phrase checks; decide from the conversation context.
- Keep reasoning short.

Latest user request: {user_request}
Chat history: {chat_history}
Latest grounded answer: {latest_answer}
"""

GROUNDED_REFORMAT_PROMPT_TEMPLATE = """
You are a grounded answer formatter.

Rewrite the grounded answer to satisfy the user's latest instruction.

Rules:
- Preserve the factual content from the grounded answer.
- Preserve inline citations exactly where claims are supported.
- Do not add new claims that are not already supported by the grounded answer.
- Follow the user's presentation instruction.
- If the grounded answer is empty, return exactly: **I don't know the answer.**

Presentation instruction: {response_instruction}
Latest user request: {user_request}
Grounded answer: {grounded_answer}
Chat history: {chat_history}
"""


def _parse_followup_interpretation(raw_content: str) -> FollowUpInterpretation:
    parsed = json.loads(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError("Follow-up interpreter returned non-object JSON")

    intent = str(parsed.get("intent") or "").strip().lower()
    if intent not in {"retrieve", "reformat"}:
        raise ValueError(f"Unsupported follow-up intent: {intent}")

    standalone_question_value = parsed.get("standalone_question")
    standalone_question = (
        str(standalone_question_value).strip()
        if isinstance(standalone_question_value, str)
        else None
    )
    if standalone_question == "":
        standalone_question = None

    response_instruction_value = parsed.get("response_instruction")
    response_instruction = (
        str(response_instruction_value).strip()
        if isinstance(response_instruction_value, str)
        else None
    )
    if response_instruction == "":
        response_instruction = None

    reasoning = str(parsed.get("reasoning") or "").strip()

    return {
        "intent": intent,
        "standalone_question": standalone_question,
        "response_instruction": response_instruction,
        "reasoning": reasoning,
    }


class FollowUpInterpreter(Runnable[State, dict[str, object | None]]):
    @override
    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object | None]:
        log_node_start("FollowUpInterpreter")
        t0 = time.perf_counter()

        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        model_id = configurable.get("model_id") or get_settings().LLM_MODEL_ID
        user_request = str(input.get("user_request") or "").strip()
        history_text = str(input.get("history_text") or "").strip()
        latest_answer = str(
            input.get("final_answer") or input.get("rag_answer") or input.get("direct_answer") or ""
        ).strip()

        prompt = PromptTemplate.from_template(FOLLOWUP_INTERPRETER_PROMPT_TEMPLATE)
        formatted = prompt.format(
            user_request=user_request,
            chat_history=history_text,
            latest_answer=latest_answer,
        )

        try:
            llm = get_llm(model_id=model_id)
            response = llm.invoke([HumanMessage(content=formatted)], config=run_config)
            raw_content = getattr(response, "content", "")
            content = raw_content.strip() if isinstance(raw_content, str) else str(raw_content)
            interpretation = _parse_followup_interpretation(content)
            duration_ms = (time.perf_counter() - t0) * 1000
            log_node_end(
                "FollowUpInterpreter",
                duration_ms=duration_ms,
                intent=interpretation["intent"],
            )
            return {
                "followup_intent": interpretation["intent"],
                "standalone_question": interpretation["standalone_question"],
                "response_instruction": interpretation["response_instruction"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("FollowUpInterpreter failed, defaulting to retrieve: %s", exc)
            duration_ms = (time.perf_counter() - t0) * 1000
            log_node_end("FollowUpInterpreter", duration_ms=duration_ms, intent="retrieve")
            return {
                "followup_intent": "retrieve",
                "standalone_question": user_request or None,
                "response_instruction": None,
            }

    @override
    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object | None]:
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)


class GroundedReformatAnswer(Runnable[State, dict[str, object]]):
    @override
    def invoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        log_node_start("GroundedReformatAnswer")
        t0 = time.perf_counter()

        run_config: RunnableConfig = cast(RunnableConfig, config or {})
        configurable = run_config.get("configurable") or {}
        model_id = configurable.get("model_id") or get_settings().LLM_MODEL_ID
        user_request = str(input.get("user_request") or "").strip()
        response_instruction = str(input.get("response_instruction") or "").strip()
        grounded_answer = str(input.get("rag_answer") or input.get("final_answer") or "").strip()
        history_text = str(input.get("history_text") or "").strip()

        if not grounded_answer:
            result_answer = "**I don't know the answer.**"
        else:
            prompt = PromptTemplate.from_template(GROUNDED_REFORMAT_PROMPT_TEMPLATE)
            formatted = prompt.format(
                response_instruction=response_instruction,
                user_request=user_request,
                grounded_answer=grounded_answer,
                chat_history=history_text,
            )
            llm = get_llm(model_id=model_id)
            response = llm.invoke([HumanMessage(content=formatted)], config=run_config)
            raw_content = getattr(response, "content", "")
            result_answer = (
                raw_content.strip() if isinstance(raw_content, str) else str(raw_content)
            )

        usage_text = messages_to_text(
            [
                HumanMessage(content=user_request),
                HumanMessage(content=response_instruction),
                HumanMessage(content=result_answer),
            ]
        )
        context_usage = calculate_context_usage(usage_text, model_id)
        log_context_usage(context_usage)

        duration_ms = (time.perf_counter() - t0) * 1000
        log_node_end(
            "GroundedReformatAnswer", duration_ms=duration_ms, answer_len=len(result_answer)
        )
        return {
            "rag_answer": result_answer,
            "rag_has_citations": "[" in result_answer and "]" in result_answer,
            "context_usage": context_usage,
        }

    @override
    async def ainvoke(
        self, input: State, config: RunnableConfig | None = None, **kwargs: object
    ) -> dict[str, object]:
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)

"""Asynchronous derived thread summaries behind a project-owned port."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UsageLimits,
    UserError,
)

from cywl_oopz.core.errors import ProviderError, ProviderResponseError, ProviderTimeoutError
from cywl_oopz.settings import AgentSettings

from .models import AgentMessage, AgentModelRef, AgentThread
from .ports import AgentMessageRepository, AgentThreadRepository
from .registry import AgentModelRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ThreadSummaryRequest:
    """Bounded input to a replaceable summary model."""

    model: AgentModelRef
    previous_summary: str
    messages: tuple[AgentMessage, ...]
    max_characters: int


class ThreadSummarizer(Protocol):
    """Generate one merged summary without mutating persistence."""

    async def summarize(self, request: ThreadSummaryRequest) -> str:
        logger.info(
            "Agent thread summarization started: model=%s/%s messages=%s",
            request.model.provider_alias,
            request.model.model_alias,
            len(request.messages),
        )
        """Return a non-empty bounded summary."""


class PydanticAiThreadSummarizer:
    """Use the run-pinned provider model for one no-tool summary request."""

    def __init__(self, registry: AgentModelRegistry, settings: AgentSettings) -> None:
        self._registry = registry
        self._settings = settings

    async def summarize(self, request: ThreadSummaryRequest) -> str:
        model = await self._registry.model(request.model)
        agent = Agent(
            model,
            instructions=(
                "你负责压缩 OOPZ 对话历史。保留用户偏好、明确事实、未完成事项、"
                "工具产生的实际结果和重要约束；不要添加原文没有的信息。"
                f"输出不超过 {request.max_characters} 个字符的纯文本中文摘要。"
            ),
        )
        transcript = self._transcript(request.messages)
        prompt = f"已有摘要：\n{request.previous_summary or '（无）'}\n\n新增对话：\n{transcript}"
        try:
            async with asyncio.timeout(self._settings.summary_timeout_seconds):
                result = await agent.run(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=1,
                        total_tokens_limit=self._settings.max_total_tokens,
                    ),
                )
        except TimeoutError as exc:
            raise ProviderTimeoutError("Agent summary timed out") from exc
        except (ModelAPIError, ModelHTTPError) as exc:
            raise ProviderError("Agent summary model request failed") from exc
        except (UnexpectedModelBehavior, UsageLimitExceeded, UserError) as exc:
            raise ProviderResponseError("Agent summary response was invalid") from exc
        output = result.output
        if not isinstance(output, str) or not output.strip():
            raise ProviderResponseError("Agent summary was empty")
        summary = output.strip()[: request.max_characters]
        logger.info("Agent thread summarization completed: summary_characters=%s", len(summary))
        return summary

    @staticmethod
    def _transcript(messages: tuple[AgentMessage, ...]) -> str:
        lines: list[str] = []
        for message in messages:
            if message.kind == "text":
                value = message.content.get("text")
            else:
                value = dict(message.content)
            lines.append(
                f"{message.role}/{message.kind}: "
                f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            )
        return "\n".join(lines)


class ThreadSummaryService:
    """Select complete old turns, generate a summary, and save it with CAS."""

    def __init__(
        self,
        settings: AgentSettings,
        summarizer: ThreadSummarizer,
        threads: AgentThreadRepository,
        messages: AgentMessageRepository,
    ) -> None:
        self._settings = settings
        self._summarizer = summarizer
        self._threads = threads
        self._messages = messages

    async def maybe_summarize(
        self,
        thread: AgentThread,
        model: AgentModelRef,
    ) -> bool:
        """Summarize only whole turns beyond the configured retention window."""
        if not self._settings.summary_enabled:
            return False
        pending = await self._messages.load_after(
            thread.id,
            after_sequence=thread.summary_through_sequence,
            limit=self._settings.summary_trigger_messages * 2,
        )
        if len(pending) < self._settings.summary_trigger_messages:
            return False
        selected = self._select_complete_turns(pending)
        if not selected:
            return False
        through_sequence = selected[-1].sequence
        if through_sequence is None:
            return False
        logger.debug(
            "Agent thread summary triggered: thread=%s selected_messages=%s through_sequence=%s",
            thread.id,
            len(selected),
            through_sequence,
        )
        summary = await self._summarizer.summarize(
            ThreadSummaryRequest(
                model=model,
                previous_summary=thread.summary,
                messages=selected,
                max_characters=self._settings.summary_max_characters,
            )
        )
        saved = await self._threads.save_summary(
            thread.id,
            summary,
            through_sequence,
            expected_version=thread.version,
        )
        logger.info(
            "Agent thread summary persistence completed: thread=%s saved=%s",
            thread.id,
            saved,
        )
        return saved

    def _select_complete_turns(
        self,
        messages: tuple[AgentMessage, ...],
    ) -> tuple[AgentMessage, ...]:
        turns: list[list[AgentMessage]] = []
        for message in messages:
            if message.role == "user":
                turns.append([])
            if turns:
                turns[-1].append(message)
        turns = [
            turn
            for turn in turns
            if turn and turn[-1].role == "assistant" and turn[-1].kind == "text"
        ]
        remaining = sum(len(turn) for turn in turns)
        selected: list[AgentMessage] = []
        for turn in turns:
            if remaining - len(turn) < self._settings.summary_retain_messages:
                break
            selected.extend(turn)
            remaining -= len(turn)
        return tuple(selected)

"""Execute one durable delegated task through the shared Agent run primitive."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from cywl_oopz.core.errors import (
    ConfigurationError,
    DatabaseError,
    ProviderError,
    ProviderSelectionError,
    ProviderTimeoutError,
)
from cywl_oopz.core.lifecycle import ModelSelectionSource, ToolEffect
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind, ProgressSink
from cywl_oopz.settings import AgentSettings

from ..catalog import ReloadableProviderCatalog
from ..context import AgentContextBuilder
from ..models import (
    AgentIdentity,
    AgentMessage,
    AgentRunLimits,
    AgentThread,
    ModelCapability,
)
from ..ports import AgentThreadRepository
from ..run_service import AgentRunService, AgentRunSpec
from ..skills.availability import SkillAvailabilityService
from ..skills.models import AgentSkillDiscovery
from ..skills.ports import AgentSkillReadRepository
from ..skills.scope import AgentSkillRunScope
from ..skills.tools import SKILL_TOOL_NAMES
from ..tools.registry import ToolRegistry
from .models import (
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskStatus,
)
from .ports import DelegatedTaskRepository, DelegatedTaskWakeup

logger = logging.getLogger(__name__)

VOICE_CONTROL_TOOL_NAMES = frozenset(
    {
        "delegate_agent_task",
        "get_agent_task",
        "list_agent_tasks",
        "read_agent_task_result",
        "cancel_agent_task",
    }
)
_WHITESPACE = re.compile(r"\s+")


class DelegatedTaskProgressSink(ProgressSink):
    """Project rich Agent progress into a bounded, throttled task status."""

    def __init__(
        self,
        repository: DelegatedTaskRepository,
        task: DelegatedAgentTask,
        worker_id: str,
        *,
        interval_seconds: float = 0.5,
    ) -> None:
        self._repository = repository
        self._task = task
        self._worker_id = worker_id
        self._interval_seconds = interval_seconds
        self._last_stage = ""
        self._last_write = 0.0
        self._lock = asyncio.Lock()

    async def emit(self, event: ConversationProgressEvent) -> None:
        stage, summary = self._projection(event)
        if not stage:
            return
        async with self._lock:
            now = time.monotonic()
            if stage == self._last_stage and now - self._last_write < self._interval_seconds:
                return
            try:
                changed = await self._repository.update_progress(
                    self._task.id,
                    self._worker_id,
                    stage,
                    summary,
                )
            except Exception as exc:
                logger.warning(
                    "Delegated task progress update failed: task=%s error=%s",
                    opaque_ref(str(self._task.id)),
                    exception_kind(exc),
                )
                return
            if changed:
                self._last_stage = stage
                self._last_write = now

    @staticmethod
    def _projection(event: ConversationProgressEvent) -> tuple[str, str]:
        if event.kind is ProgressKind.MODEL_RETRY:
            return "model_retry", f"上游重试 {event.retry_attempt}/{event.retry_max_attempts}"
        if event.kind in {
            ProgressKind.TOOL_STARTED,
            ProgressKind.TOOL_UPDATED,
            ProgressKind.TOOL_SUCCEEDED,
            ProgressKind.TOOL_FAILED,
        }:
            detail = event.tool_summary or event.tool_subject
            return event.kind.value, _line(
                " · ".join(item for item in (event.tool_display_name, detail) if item),
                512,
            )
        if event.kind in {ProgressKind.THINKING, ProgressKind.TEXT_RESET}:
            return "thinking", "正在分析任务"
        if event.kind is ProgressKind.TEXT_DELTA:
            return "writing", "正在整理结果"
        return "", ""


class DelegatedAgentTaskRunner:
    """Build an isolated task context and persist one deterministic task outcome."""

    def __init__(
        self,
        settings: AgentSettings,
        repository: DelegatedTaskRepository,
        wakeup: DelegatedTaskWakeup,
        run_service: AgentRunService,
        catalog: ReloadableProviderCatalog,
        threads: AgentThreadRepository,
        context_builder: AgentContextBuilder,
        tools: ToolRegistry,
        skill_repository: AgentSkillReadRepository | None = None,
        skill_availability: SkillAvailabilityService | None = None,
        *,
        max_task_retries: int = 2,
        heartbeat_interval_seconds: float = 10.0,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_task_retries < 0:
            raise ValueError("Delegated task retries must not be negative")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("Delegated task heartbeat interval must be positive")
        self._settings = settings
        self._repository = repository
        self._wakeup = wakeup
        self._run_service = run_service
        self._catalog = catalog
        self._threads = threads
        self._context_builder = context_builder
        self._tools = tools
        self._skill_repository = skill_repository
        self._skill_availability = skill_availability
        self._max_task_retries = max_task_retries
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._jitter = jitter

    async def run(self, task: DelegatedAgentTask, worker_id: str) -> None:
        task_ref = opaque_ref(str(task.id))
        logger.info(
            "Delegated Agent task started: task=%s alias=%s lane=%s retry=%s",
            task_ref,
            task.alias,
            task.lane.value,
            task.retry_count,
        )
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._heartbeat(task, worker_id, owner_task),
            name=f"delegated-heartbeat:{task.id}",
        )
        try:
            await self._execute(task, worker_id)
        except asyncio.CancelledError:
            logger.info("Delegated Agent task execution cancelled: task=%s", task_ref)
            raise
        except ProviderSelectionError as exc:
            await self._fail(task, worker_id, "model_unavailable", "后台模型当前不可用", exc)
        except ProviderTimeoutError as exc:
            await self._retry_or_fail(
                task,
                worker_id,
                "provider_timeout",
                "上游模型请求超时",
                exc,
            )
        except ProviderError as exc:
            await self._retry_or_fail(
                task,
                worker_id,
                "provider_error",
                "上游模型请求失败",
                exc,
            )
        except ConfigurationError as exc:
            await self._fail(task, worker_id, "invalid_configuration", "后台配置无效", exc)
        except DatabaseError as exc:
            await self._retry_or_fail(
                task,
                worker_id,
                "database_error",
                "后台存储暂时不可用",
                exc,
            )
        except Exception as exc:
            await self._fail(task, worker_id, "agent_error", "后台任务执行失败", exc)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _execute(self, task: DelegatedAgentTask, worker_id: str) -> None:
        enabled_tools = self._enabled_tools(task)
        await self._catalog.reload()
        model = self._catalog.snapshot.resolve(
            task.agent_model_id,
            required_capabilities=(
                frozenset({ModelCapability.TOOL_CALLING}) if enabled_tools else frozenset()
            ),
            require_user_selectable=False,
        )
        if model is None:
            raise ProviderSelectionError("Pinned delegated Agent model is unavailable")

        thread = await self._task_thread(task)
        identity = AgentIdentity(
            task.owner_person_id,
            thread.key,
            transport_channel_id=task.text_channel_id,
        )
        available_skills: tuple[AgentSkillDiscovery, ...] = ()
        skill_scope: AgentSkillRunScope | None = None
        if self._skill_repository is not None and self._skill_availability is not None:
            discoveries = await self._skill_repository.list_accessible(task.owner_person_id)
            available_skills = self._skill_availability.resolve(discoveries, enabled_tools)
            if available_skills:
                skill_scope = AgentSkillRunScope(
                    self._skill_repository,
                    task.owner_person_id,
                    available_skills,
                    max_activations=self._settings.max_skill_activations,
                    max_resources=self._settings.max_skill_resources,
                    max_instruction_characters=self._settings.max_skill_instruction_characters,
                    max_resource_characters=self._settings.max_skill_resource_characters,
                    max_context_characters=self._settings.max_skill_context_characters,
                )
            else:
                enabled_tools = tuple(
                    name for name in enabled_tools if name not in SKILL_TOOL_NAMES
                )

        context = await self._context_builder.build(
            thread,
            identity,
            available_skills=available_skills,
            include_history=False,
        )
        context = (
            context[0],
            AgentMessage(
                "system",
                "delegated_task",
                {"text": self._task_instructions(task.result_style)},
            ),
            *context[1:],
        )
        outcome = await self._run_service.run(
            AgentRunSpec(
                thread=thread,
                identity=identity,
                prompt=task.objective,
                model=model,
                selection_source=ModelSelectionSource.APPLICATION,
                enabled_tools=enabled_tools,
                limits=self._run_limits(),
                context=context,
                skill_scope=skill_scope,
            ),
            DelegatedTaskProgressSink(self._repository, task, worker_id),
        )
        result_text = _bounded_text(outcome.result.output, 16_000)
        completed = await self._repository.complete(
            task.id,
            worker_id,
            _result_summary(result_text),
            result_text,
            agent_thread_id=thread.id,
            agent_run_id=outcome.run_id,
        )
        if not completed:
            await self._finish_cancel_race(task.id, worker_id)
            return
        logger.info(
            "Delegated Agent task completed: task=%s elapsed_seconds=%.3f tools=%s",
            opaque_ref(str(task.id)),
            outcome.elapsed_seconds,
            outcome.result.tool_calls,
        )
        await self._wakeup.wake(task.id)

    def _enabled_tools(self, task: DelegatedAgentTask) -> tuple[str, ...]:
        if VOICE_CONTROL_TOOL_NAMES.intersection(task.allowed_tool_names):
            logger.warning(
                "Delegated task contained recursive control tools: task=%s",
                opaque_ref(str(task.id)),
            )
        descriptors = {
            descriptor.name: descriptor
            for descriptor in self._tools.descriptors(task.allowed_tool_names)
        }
        allowed = tuple(
            name
            for name in task.allowed_tool_names
            if name in descriptors
            and name not in VOICE_CONTROL_TOOL_NAMES
            and (
                task.lane is not DelegatedTaskLane.READ_PARALLEL
                or descriptors[name].effect is ToolEffect.READ
            )
        )
        if task.allowed_tool_names and not allowed:
            raise ConfigurationError("No pinned delegated Agent tools are registered")
        return allowed

    async def _task_thread(self, task: DelegatedAgentTask) -> AgentThread:
        key = ConversationKey(
            "delegated_task",
            task.area_id,
            str(task.id),
            task.owner_person_id,
        )
        existing = await self._threads.get(key)
        if existing is not None:
            return existing
        thread = AgentThread(
            id=uuid4(),
            key=key,
            selected_model_id=task.agent_model_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=self._settings.session_ttl_seconds),
        )
        await self._threads.add(thread)
        return thread

    async def _retry_or_fail(
        self,
        task: DelegatedAgentTask,
        worker_id: str,
        error_code: str,
        safe_message: str,
        error: Exception,
    ) -> None:
        if (
            task.lane is DelegatedTaskLane.READ_PARALLEL
            and task.retry_count < self._max_task_retries
        ):
            base_delay = min(30.0, 2.0**task.retry_count)
            next_attempt = datetime.now(UTC) + timedelta(
                seconds=base_delay + self._jitter(0.0, base_delay * 0.25)
            )
            scheduled = await self._repository.mark_waiting_retry(
                task.id,
                worker_id,
                next_attempt,
                error_code,
                safe_message,
            )
            if not scheduled:
                await self._finish_cancel_race(task.id, worker_id)
                return
            logger.warning(
                "Delegated Agent task retry scheduled: task=%s retry=%s/%s error=%s",
                opaque_ref(str(task.id)),
                task.retry_count + 1,
                self._max_task_retries,
                exception_kind(error),
            )
            await self._wakeup.wake(task.id)
            return
        await self._fail(task, worker_id, error_code, safe_message, error)

    async def _fail(
        self,
        task: DelegatedAgentTask,
        worker_id: str,
        error_code: str,
        safe_message: str,
        error: Exception,
    ) -> None:
        failed = await self._repository.fail(
            task.id,
            worker_id,
            error_code,
            safe_message,
        )
        if not failed:
            await self._finish_cancel_race(task.id, worker_id)
            return
        logger.warning(
            "Delegated Agent task failed: task=%s code=%s error=%s",
            opaque_ref(str(task.id)),
            error_code,
            exception_kind(error),
        )
        await self._wakeup.wake(task.id)

    async def _finish_cancel_race(self, task_id, worker_id: str) -> None:
        current = await self._repository.get(task_id)
        if current is not None and current.status is DelegatedTaskStatus.CANCEL_REQUESTED:
            await self._repository.mark_cancelled(task_id, worker_id)
            await self._wakeup.wake(task_id)

    async def _heartbeat(
        self,
        task: DelegatedAgentTask,
        worker_id: str,
        owner_task: asyncio.Task[object] | None,
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                alive = await self._repository.heartbeat(task.id, worker_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Delegated task heartbeat failed: task=%s error=%s",
                    opaque_ref(str(task.id)),
                    exception_kind(exc),
                )
                continue
            if not alive:
                logger.warning(
                    "Delegated task lost worker ownership: task=%s",
                    opaque_ref(str(task.id)),
                )
                if owner_task is not None:
                    owner_task.cancel()
                return

    def _run_limits(self) -> AgentRunLimits:
        return AgentRunLimits(
            timeout_seconds=self._settings.timeout_seconds,
            max_model_requests=self._settings.max_model_requests,
            max_tool_calls=self._settings.max_tool_calls,
            max_total_tokens=self._settings.max_total_tokens,
            max_parallel_tools=self._settings.max_parallel_tools,
        )

    @staticmethod
    def _task_instructions(style: DelegatedResultStyle) -> str:
        detail = (
            "给出较完整但有界的结果，保留关键依据和实际使用的来源 URL。"
            if style is DelegatedResultStyle.DETAILED
            else "给出 1 至 3 句可直接播报的摘要，必要时附最关键的来源 URL。"
        )
        return (
            "你正在执行一个由实时语音会话委派的独立后台任务。"
            "只完成当前用户目标，不创建或调用新的语音委派任务，也不要假装仍在实时语音中。"
            "可用工具结果是事实依据；工具不足或失败时明确说明。"
            f"{detail} 最终只输出给用户看的结果，不输出内部过程、任务状态或等待话术。"
        )


def _line(value: str, limit: int) -> str:
    return _WHITESPACE.sub(" ", value).strip()[:limit]


def _bounded_text(value: str, limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _result_summary(value: str) -> str:
    first_paragraph = next((part.strip() for part in value.splitlines() if part.strip()), value)
    return _bounded_text(_line(first_paragraph, 1000), 1000)

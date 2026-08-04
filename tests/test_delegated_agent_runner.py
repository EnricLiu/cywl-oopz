from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cywl_oopz.core.errors import ProviderError
from cywl_oopz.core.lifecycle import AgentStopReason, ToolEffect
from cywl_oopz.features.agent.catalog import ProviderCatalog
from cywl_oopz.features.agent.delegation.models import (
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskStatus,
)
from cywl_oopz.features.agent.delegation.runner import DelegatedAgentTaskRunner
from cywl_oopz.features.agent.models import (
    AgentMessage,
    AgentRunResult,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.run_service import AgentRunOutcome
from cywl_oopz.features.agent.skills.availability import SkillAvailabilityService
from cywl_oopz.features.agent.skills.models import AgentSkillDiscovery, SkillAccessKind
from cywl_oopz.settings import AppSettings


class MemoryRunnerRepository:
    def __init__(self, task: DelegatedAgentTask) -> None:
        self.task = task
        self.progress: list[tuple[str, str]] = []
        self.retry_at: datetime | None = None

    async def heartbeat(self, task_id: UUID, worker_id: str) -> bool:
        del task_id, worker_id
        return True

    async def update_progress(self, task_id, worker_id, stage, summary) -> bool:
        del task_id, worker_id
        self.progress.append((stage, summary))
        return True

    async def complete(
        self,
        task_id,
        worker_id,
        result_summary,
        result_text,
        *,
        agent_thread_id=None,
        agent_run_id=None,
    ) -> bool:
        del task_id, worker_id
        self.task = replace(
            self.task,
            status=DelegatedTaskStatus.SUCCEEDED,
            result_summary=result_summary,
            result_text=result_text,
            agent_thread_id=agent_thread_id,
            agent_run_id=agent_run_id,
        )
        return True

    async def mark_waiting_retry(
        self,
        task_id,
        worker_id,
        next_attempt_at,
        error_code,
        error_message="",
    ) -> bool:
        del task_id, worker_id
        self.retry_at = next_attempt_at
        self.task = replace(
            self.task,
            status=DelegatedTaskStatus.WAITING_RETRY,
            retry_count=self.task.retry_count + 1,
            error_code=error_code,
            error_message=error_message,
        )
        return True

    async def fail(self, task_id, worker_id, error_code, error_message="") -> bool:
        del task_id, worker_id
        self.task = replace(
            self.task,
            status=DelegatedTaskStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )
        return True

    async def get(self, task_id):
        return self.task if self.task.id == task_id else None

    async def mark_cancelled(self, task_id, worker_id) -> bool:
        del task_id, worker_id
        self.task = replace(self.task, status=DelegatedTaskStatus.CANCELLED)
        return True


class RecordingWakeup:
    def __init__(self) -> None:
        self.ids: list[UUID] = []

    async def wake(self, task_id: UUID) -> None:
        self.ids.append(task_id)


class FakeCatalog:
    def __init__(self, catalog: ProviderCatalog) -> None:
        self.snapshot = catalog
        self.reloads = 0

    async def reload(self) -> ProviderCatalog:
        self.reloads += 1
        return self.snapshot


class MemoryThreads:
    def __init__(self) -> None:
        self.value = None

    async def get(self, key):
        return self.value if self.value is not None and self.value.key == key else None

    async def add(self, thread) -> None:
        self.value = thread


class RecordingContextBuilder:
    def __init__(self) -> None:
        self.calls = []

    async def build(self, thread, identity, **kwargs):
        self.calls.append((thread, identity, kwargs))
        return (AgentMessage("system", "text", {"text": "base"}),)


class NamedTools:
    names = (
        "search_web",
        "load_agent_skill",
        "read_agent_skill_resource",
        "dangerous_write",
        "delegate_agent_task",
    )

    def descriptors(self, names=None):
        allowed = frozenset(names or self.names)
        return tuple(
            SimpleNamespace(
                name=name,
                effect=(ToolEffect.WRITE if name == "dangerous_write" else ToolEffect.READ),
            )
            for name in self.names
            if name in allowed
        )


class AccessibleSkills:
    def __init__(self) -> None:
        self.discovery = AgentSkillDiscovery(
            uuid4(),
            "research-sources",
            "Research sources",
            "查找并核对公开来源",
            "1.0.0",
            1,
            frozenset({"search_web"}),
            SkillAccessKind.OWNED,
        )

    async def list_accessible(self, person_id: str):
        assert person_id == "person"
        return (self.discovery,)


class SuccessfulRunService:
    def __init__(self) -> None:
        self.spec = None

    async def run(self, spec, progress=None):
        self.spec = spec
        if progress is not None:
            from cywl_oopz.features.chat.progress import (
                ConversationProgressEvent,
                ProgressKind,
            )

            await progress.emit(ConversationProgressEvent(ProgressKind.THINKING))
        return AgentRunOutcome(
            uuid4(),
            AgentRunResult(
                "找到三条可靠结果。\nhttps://example.com",
                AgentStopReason.COMPLETED,
                tool_calls=1,
            ),
            0.1,
        )


class FailingRunService:
    async def run(self, spec, progress=None):
        del spec, progress
        raise ProviderError("transport")


def settings():
    return AppSettings.from_mapping(
        {
            "OOPZ_DEVICE_ID": "device",
            "OOPZ_PERSON_UID": "bot",
            "OOPZ_JWT_TOKEN": "token",
            "DATABASE_URL": "postgresql://user:secret@localhost/cywl",
            "CYWL_CHAT_ENABLED": "false",
            "CYWL_AGENT_MODE": "agent",
        }
    ).agent


def task(model_id: UUID) -> DelegatedAgentTask:
    return DelegatedAgentTask(
        id=uuid4(),
        owner_person_id="person",
        area_id="area",
        text_channel_id="text",
        voice_channel_id="voice",
        origin_voice_session_id=uuid4(),
        session_sequence=1,
        provider_call_id="call",
        objective="搜索最近的公开信息",
        result_style=DelegatedResultStyle.BRIEF,
        status=DelegatedTaskStatus.RUNNING,
        lane=DelegatedTaskLane.READ_PARALLEL,
        conflict_key="",
        notification_state=DelegatedTaskNotificationState.PENDING,
        agent_model_id=model_id,
        allowed_tool_names=(
            "search_web",
            "load_agent_skill",
            "read_agent_skill_resource",
            "dangerous_write",
            "delegate_agent_task",
        ),
    )


def catalog() -> tuple[UUID, ProviderCatalog]:
    provider_id = uuid4()
    model_id = uuid4()
    provider = LlmProvider(
        provider_id,
        "provider",
        "Provider",
        ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        "https://llm.example/v1",
        "key",
        True,
        True,
    )
    model = LlmModel(
        model_id,
        provider_id,
        "model",
        "remote-model",
        "Model",
        True,
        True,
        True,
        frozenset({ModelCapability.TOOL_CALLING}),
    )
    return model_id, ProviderCatalog.build((provider,), (model,))


@pytest.mark.asyncio
async def test_runner_uses_isolated_context_and_never_exposes_recursive_tools() -> None:
    model_id, provider_catalog = catalog()
    envelope = task(model_id)
    repository = MemoryRunnerRepository(envelope)
    wakeup = RecordingWakeup()
    run_service = SuccessfulRunService()
    context = RecordingContextBuilder()
    skills = AccessibleSkills()
    runner = DelegatedAgentTaskRunner(
        settings(),
        repository,
        wakeup,
        run_service,
        FakeCatalog(provider_catalog),
        MemoryThreads(),
        context,
        NamedTools(),
        skills,
        SkillAvailabilityService(),
        max_task_retries=2,
        heartbeat_interval_seconds=10,
    )

    await runner.run(envelope, "worker")

    assert repository.task.status is DelegatedTaskStatus.SUCCEEDED
    assert repository.task.result_summary == "找到三条可靠结果。"
    assert repository.task.agent_thread_id is not None
    assert repository.task.agent_run_id is not None
    assert run_service.spec.identity.conversation.scope == "delegated_task"
    assert run_service.spec.identity.conversation.channel_id == str(envelope.id)
    assert run_service.spec.identity.source_message_id == ""
    assert run_service.spec.identity.transport_channel_id == "text"
    assert run_service.spec.enabled_tools == (
        "search_web",
        "load_agent_skill",
        "read_agent_skill_resource",
    )
    assert run_service.spec.skill_scope is not None
    assert run_service.spec.skill_scope.available_skills == (skills.discovery,)
    assert "独立后台任务" in run_service.spec.context[1].content["text"]
    assert context.calls[0][2]["include_history"] is False
    assert context.calls[0][2]["available_skills"] == (skills.discovery,)
    assert repository.progress == [("thinking", "正在分析任务")]
    assert wakeup.ids == [envelope.id]


@pytest.mark.asyncio
async def test_runner_schedules_task_retry_without_sleeping_worker() -> None:
    model_id, provider_catalog = catalog()
    envelope = task(model_id)
    repository = MemoryRunnerRepository(envelope)
    wakeup = RecordingWakeup()
    runner = DelegatedAgentTaskRunner(
        settings(),
        repository,
        wakeup,
        FailingRunService(),
        FakeCatalog(provider_catalog),
        MemoryThreads(),
        RecordingContextBuilder(),
        NamedTools(),
        max_task_retries=2,
        heartbeat_interval_seconds=10,
        jitter=lambda low, high: 0.0,
    )

    started = datetime.now(UTC)
    await asyncio.wait_for(runner.run(envelope, "worker"), timeout=0.1)

    assert repository.task.status is DelegatedTaskStatus.WAITING_RETRY
    assert repository.task.retry_count == 1
    assert repository.task.error_code == "provider_error"
    assert repository.retry_at is not None
    assert 0.9 <= (repository.retry_at - started).total_seconds() <= 1.2
    assert wakeup.ids == [envelope.id]

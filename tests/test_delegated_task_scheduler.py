from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cywl_oopz.features.agent.delegation.models import (
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskStatus,
    RecoverySummary,
)
from cywl_oopz.features.agent.delegation.scheduler import DelegatedTaskScheduler
from cywl_oopz.features.agent.delegation.service import InProcessDelegatedTaskWakeup
from cywl_oopz.features.voice.models import VoiceStartRequest, VoiceTextAddress
from cywl_oopz.features.voice.service import VoiceConversationService
from cywl_oopz.integrations.voice.fake import (
    FakeVoiceAccessGateway,
    FakeVoiceConfigurationRepository,
    FakeVoiceSessionRepository,
    FakeVoiceSessionRuntimeFactory,
)
from cywl_oopz.settings import VoiceSettings


class MemorySchedulerRepository:
    def __init__(self, tasks: list[DelegatedAgentTask]) -> None:
        self.tasks = {task.id: task for task in tasks}
        self.recovery_calls = 0

    async def recover_stale(self, now: datetime) -> RecoverySummary:
        self.recovery_calls += 1
        changed: list[UUID] = []
        requeued = cancelled = interrupted = 0
        for task_id, task in tuple(self.tasks.items()):
            if task.status is DelegatedTaskStatus.CANCEL_REQUESTED:
                self.tasks[task_id] = replace(
                    task,
                    status=DelegatedTaskStatus.CANCELLED,
                    finished_at=now,
                )
                cancelled += 1
                changed.append(task_id)
            elif task.status is DelegatedTaskStatus.RUNNING:
                if task.lane is DelegatedTaskLane.READ_PARALLEL:
                    self.tasks[task_id] = replace(
                        task,
                        status=DelegatedTaskStatus.QUEUED,
                        retry_count=task.retry_count + 1,
                    )
                    requeued += 1
                else:
                    self.tasks[task_id] = replace(
                        task,
                        status=DelegatedTaskStatus.INTERRUPTED,
                        finished_at=now,
                    )
                    interrupted += 1
                changed.append(task_id)
        return RecoverySummary(requeued, cancelled, interrupted, tuple(changed))

    async def claim_next(
        self,
        worker_id: str,
        lanes: frozenset[DelegatedTaskLane],
        *,
        excluded_owner_person_ids: frozenset[str] = frozenset(),
    ) -> DelegatedAgentTask | None:
        for task_id, task in self.tasks.items():
            if (
                task.status in {DelegatedTaskStatus.QUEUED, DelegatedTaskStatus.WAITING_RETRY}
                and task.lane in lanes
                and task.owner_person_id not in excluded_owner_person_ids
            ):
                claimed = replace(task, status=DelegatedTaskStatus.RUNNING)
                self.tasks[task_id] = claimed
                return claimed
        return None

    async def get(self, task_id: UUID) -> DelegatedAgentTask | None:
        return self.tasks.get(task_id)

    async def mark_cancelled(self, task_id: UUID, worker_id: str) -> bool:
        del worker_id
        task = self.tasks[task_id]
        if task.status not in {
            DelegatedTaskStatus.RUNNING,
            DelegatedTaskStatus.CANCEL_REQUESTED,
        }:
            return False
        self.tasks[task_id] = replace(
            task,
            status=DelegatedTaskStatus.CANCELLED,
            finished_at=datetime.now(UTC),
        )
        return True

    def request_running_cancel(self, task_id: UUID) -> None:
        self.tasks[task_id] = replace(
            self.tasks[task_id],
            status=DelegatedTaskStatus.CANCEL_REQUESTED,
        )


class BlockingRunner:
    def __init__(self) -> None:
        self.started: list[UUID] = []
        self.cancelled: list[UUID] = []
        self.finished: list[UUID] = []
        self._releases: dict[UUID, asyncio.Event] = {}

    async def run(self, task: DelegatedAgentTask, worker_id: str) -> None:
        del worker_id
        self.started.append(task.id)
        release = self._releases.setdefault(task.id, asyncio.Event())
        try:
            await release.wait()
            self.finished.append(task.id)
        except asyncio.CancelledError:
            self.cancelled.append(task.id)
            raise

    def release(self, task_id: UUID) -> None:
        self._releases[task_id].set()


class RecordingCompletionNotifier:
    def __init__(self) -> None:
        self.owners: list[str] = []

    async def wake(self, owner_person_id: str) -> None:
        self.owners.append(owner_person_id)


def task(owner: str, *, status=DelegatedTaskStatus.QUEUED) -> DelegatedAgentTask:
    now = datetime.now(UTC)
    return DelegatedAgentTask(
        id=uuid4(),
        owner_person_id=owner,
        area_id="area",
        text_channel_id="text",
        voice_channel_id="voice",
        origin_voice_session_id=uuid4(),
        session_sequence=1,
        provider_call_id=str(uuid4()),
        objective="查找公开信息",
        result_style=DelegatedResultStyle.BRIEF,
        status=status,
        lane=DelegatedTaskLane.READ_PARALLEL,
        conflict_key="",
        notification_state=DelegatedTaskNotificationState.PENDING,
        agent_model_id=uuid4(),
        allowed_tool_names=("search_web",),
        created_at=now,
        updated_at=now,
    )


async def eventually(predicate, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_scheduler_runs_different_users_in_parallel_but_serializes_one_user() -> None:
    first = task("owner-a")
    second = task("owner-a")
    other = task("owner-b")
    repository = MemorySchedulerRepository([first, second, other])
    wakeup = InProcessDelegatedTaskWakeup()
    runner = BlockingRunner()
    scheduler = DelegatedTaskScheduler(
        repository,
        wakeup,
        runner,
        read_concurrency=2,
        per_user_concurrency=1,
        reconcile_seconds=0.02,
        worker_id="test-worker",
    )

    await scheduler.start()
    await eventually(lambda: len(runner.started) == 2)
    assert set(runner.started) == {first.id, other.id}
    assert second.id not in runner.started

    runner.release(other.id)
    await asyncio.sleep(0.05)
    assert second.id not in runner.started
    runner.release(first.id)
    await eventually(lambda: second.id in runner.started)

    await scheduler.aclose()
    assert repository.tasks[second.id].status is DelegatedTaskStatus.QUEUED


@pytest.mark.asyncio
async def test_scheduler_recovers_restart_and_cancels_a_running_task() -> None:
    recovered = task("owner", status=DelegatedTaskStatus.RUNNING)
    repository = MemorySchedulerRepository([recovered])
    wakeup = InProcessDelegatedTaskWakeup()
    runner = BlockingRunner()
    completion_notifier = RecordingCompletionNotifier()
    scheduler = DelegatedTaskScheduler(
        repository,
        wakeup,
        runner,
        completion_notifier=completion_notifier,
        reconcile_seconds=0.02,
        worker_id="test-worker",
    )

    await scheduler.start()
    await eventually(lambda: recovered.id in runner.started)
    assert repository.recovery_calls == 1
    assert repository.tasks[recovered.id].retry_count == 1

    repository.request_running_cancel(recovered.id)
    await wakeup.wake(recovered.id)
    await eventually(lambda: repository.tasks[recovered.id].status is DelegatedTaskStatus.CANCELLED)
    assert recovered.id in runner.cancelled
    assert completion_notifier.owners == ["owner"]

    await scheduler.aclose()
    assert repository.tasks[recovered.id].status is DelegatedTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_scheduler_serializes_mutations_with_the_same_resource_key() -> None:
    first = replace(
        task("owner-a"),
        lane=DelegatedTaskLane.MUTATION_SERIAL,
        conflict_key="area:shared",
    )
    second = replace(
        task("owner-b"),
        lane=DelegatedTaskLane.MUTATION_SERIAL,
        conflict_key="area:shared",
    )
    repository = MemorySchedulerRepository([first, second])
    wakeup = InProcessDelegatedTaskWakeup()
    runner = BlockingRunner()
    scheduler = DelegatedTaskScheduler(
        repository,
        wakeup,
        runner,
        reconcile_seconds=0.02,
        worker_id="test-worker",
    )

    await scheduler.start()
    await eventually(lambda: first.id in runner.started)
    assert second.id not in runner.started

    runner.release(first.id)
    await eventually(lambda: second.id in runner.started)
    assert runner.started == [first.id, second.id]

    await scheduler.aclose()


@pytest.mark.asyncio
async def test_voice_stop_does_not_cancel_an_accepted_background_task() -> None:
    envelope = task("owner")
    repository = MemorySchedulerRepository([envelope])
    wakeup = InProcessDelegatedTaskWakeup()
    runner = BlockingRunner()
    scheduler = DelegatedTaskScheduler(
        repository,
        wakeup,
        runner,
        reconcile_seconds=0.02,
        worker_id="test-worker",
    )
    access = FakeVoiceAccessGateway()
    access.channels[("area", "owner")] = "voice"
    conversations = VoiceConversationService(
        VoiceSettings.from_mapping(
            {
                "CYWL_VOICE_ENABLED": "true",
                "CYWL_VOICE_START_TIMEOUT_SECONDS": "1",
            }
        ),
        access,
        FakeVoiceSessionRuntimeFactory(),
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )

    await scheduler.start()
    await eventually(lambda: envelope.id in runner.started)
    await conversations.start(VoiceStartRequest("owner", VoiceTextAddress("area", "text")))

    await conversations.stop("owner")

    assert envelope.id not in runner.cancelled
    assert scheduler.active_count == 1
    runner.release(envelope.id)
    await eventually(lambda: envelope.id in runner.finished)

    await conversations.aclose()
    await scheduler.aclose()

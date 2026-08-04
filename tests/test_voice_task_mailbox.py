from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from cywl_oopz.features.agent.delegation.mailbox import (
    DelegatedTaskTextFallbackReconciler,
    InProcessVoiceTaskCompletionNotifier,
    VoiceTaskMailboxService,
    render_task_notifications,
)
from cywl_oopz.features.agent.delegation.models import (
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskStatus,
)
from cywl_oopz.features.voice.models import (
    VoiceTaskNotification,
    VoiceTaskNotificationStatus,
    VoiceTextAddress,
)


class MemoryNotificationRepository:
    def __init__(self, tasks: list[DelegatedAgentTask] | None = None) -> None:
        self.tasks = list(tasks or [])
        self.recovered = 0

    async def claim_notifications(self, session_id: UUID, limit: int):
        return self._claim(
            lambda task: task.origin_voice_session_id == session_id,
            limit,
        )

    async def claim_text_notifications(self, limit: int):
        return self._claim(lambda task: True, limit)

    def _claim(self, predicate, limit: int):
        claimed = []
        for index, task in enumerate(self.tasks):
            if len(claimed) >= limit:
                break
            if predicate(task) and task.notification_state in {
                DelegatedTaskNotificationState.PENDING,
                DelegatedTaskNotificationState.DEFERRED,
            }:
                task = replace(task, notification_state=DelegatedTaskNotificationState.CLAIMED)
                self.tasks[index] = task
                claimed.append(task)
        return tuple(claimed)

    async def mark_presented(self, task_ids: tuple[UUID, ...]) -> None:
        self._transition(task_ids, DelegatedTaskNotificationState.PRESENTED)

    async def defer_notifications(self, task_ids: tuple[UUID, ...]) -> None:
        self._transition(task_ids, DelegatedTaskNotificationState.DEFERRED)

    async def recover_claimed_notifications(self) -> int:
        claimed = tuple(
            task.id
            for task in self.tasks
            if task.notification_state is DelegatedTaskNotificationState.CLAIMED
        )
        self._transition(claimed, DelegatedTaskNotificationState.DEFERRED)
        self.recovered += len(claimed)
        return len(claimed)

    def _transition(
        self,
        task_ids: tuple[UUID, ...],
        state: DelegatedTaskNotificationState,
    ) -> None:
        selected = set(task_ids)
        self.tasks = [
            replace(task, notification_state=state) if task.id in selected else task
            for task in self.tasks
        ]


class RecordingTextGateway:
    def __init__(self) -> None:
        self.messages: list[tuple[VoiceTextAddress, str]] = []
        self.error: Exception | None = None

    async def send(self, address: VoiceTextAddress, text: str) -> None:
        if self.error is not None:
            raise self.error
        self.messages.append((address, text))


@pytest.mark.asyncio
async def test_mailbox_groups_terminal_tasks_and_commits_delivery_state() -> None:
    session_id = uuid4()
    repository = MemoryNotificationRepository(
        [
            task(session_id, 1, DelegatedTaskStatus.SUCCEEDED, summary="找到三条结果"),
            task(session_id, 2, DelegatedTaskStatus.FAILED, error="网页响应超时"),
        ]
    )
    signal = InProcessVoiceTaskCompletionNotifier()
    gateway = RecordingTextGateway()
    mailbox = VoiceTaskMailboxService(repository, signal, gateway)

    await signal.wake("owner")
    assert await mailbox.wait("owner", 0.1) is True
    notices = await mailbox.claim(session_id, 3)
    assert [notice.alias for notice in notices] == ["T1", "T2"]
    assert await mailbox.present_text(notices) is True

    assert gateway.messages == [
        (
            VoiceTextAddress("area", "text"),
            "✅ **后台任务 T1** 查询任务 1 · 找到三条结果\n"
            "⚠️ **后台任务 T2** 查询任务 2 · 网页响应超时",
        )
    ]
    assert all(
        item.notification_state is DelegatedTaskNotificationState.PRESENTED
        for item in repository.tasks
    )


@pytest.mark.asyncio
async def test_mailbox_defers_claimed_tasks_when_text_delivery_fails() -> None:
    session_id = uuid4()
    repository = MemoryNotificationRepository([task(session_id, 1)])
    gateway = RecordingTextGateway()
    gateway.error = RuntimeError("fixture send failure")
    mailbox = VoiceTaskMailboxService(
        repository,
        InProcessVoiceTaskCompletionNotifier(),
        gateway,
    )

    notices = await mailbox.claim(session_id, 3)
    assert await mailbox.present_text(notices) is False
    assert repository.tasks[0].notification_state is DelegatedTaskNotificationState.DEFERRED


@pytest.mark.asyncio
async def test_text_fallback_recovers_claims_and_coalesces_three_per_message() -> None:
    session_id = uuid4()
    tasks = [task(session_id, index) for index in range(1, 5)]
    tasks[0] = replace(tasks[0], notification_state=DelegatedTaskNotificationState.CLAIMED)
    repository = MemoryNotificationRepository(tasks)
    signal = InProcessVoiceTaskCompletionNotifier()
    gateway = RecordingTextGateway()
    mailbox = VoiceTaskMailboxService(repository, signal, gateway)
    reconciler = DelegatedTaskTextFallbackReconciler(
        repository,
        signal,
        mailbox,
        poll_seconds=30,
        group_limit=3,
    )

    await reconciler.start()
    await wait_until(lambda: len(gateway.messages) == 2)
    await reconciler.aclose()

    assert repository.recovered == 1
    assert gateway.messages[0][1].count("**后台任务") == 3
    assert gateway.messages[1][1].count("**后台任务") == 1
    assert all(
        item.notification_state is DelegatedTaskNotificationState.PRESENTED
        for item in repository.tasks
    )


def test_notification_renderer_normalizes_and_bounds_dynamic_text() -> None:
    notice = VoiceTaskNotification(
        uuid4(),
        "T7",
        VoiceTaskNotificationStatus.SUCCEEDED,
        "  查询\n初音未来   新闻  ",
        "结果 " + "很长" * 100,
        "",
        VoiceTextAddress("area", "text"),
    )

    rendered = render_task_notifications((notice,))

    assert rendered.startswith("✅ **后台任务 T7** 查询 初音未来 新闻 · 结果 ")
    assert rendered.endswith("…")
    assert "\n" not in rendered


def task(
    session_id: UUID,
    sequence: int,
    status: DelegatedTaskStatus = DelegatedTaskStatus.SUCCEEDED,
    *,
    summary: str = "任务已完成",
    error: str = "",
) -> DelegatedAgentTask:
    return DelegatedAgentTask(
        uuid4(),
        "owner",
        "area",
        "text",
        "voice",
        session_id,
        sequence,
        f"call-{sequence}",
        f"查询任务 {sequence}",
        DelegatedResultStyle.BRIEF,
        status,
        DelegatedTaskLane.READ_PARALLEL,
        "",
        DelegatedTaskNotificationState.PENDING,
        uuid4(),
        ("search_web",),
        result_summary=summary,
        error_message=error,
    )


async def wait_until(predicate, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")

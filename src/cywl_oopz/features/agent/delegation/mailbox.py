"""Terminal task mailbox, text fallback, and lossy completion signals."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Protocol
from uuid import UUID

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.voice.models import (
    VoiceTaskNotification,
    VoiceTaskNotificationStatus,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceTaskMailbox

from .models import DelegatedAgentTask
from .ports import DelegatedTaskCompletionNotifier, DelegatedTaskRepository

logger = logging.getLogger(__name__)
_WHITESPACE = re.compile(r"\s+")


class VoiceTaskTextGateway(Protocol):
    async def send(self, address: VoiceTextAddress, text: str) -> None: ...


class VoiceTaskCompletionSignal(DelegatedTaskCompletionNotifier, Protocol):
    async def wait_owner(self, owner_person_id: str, timeout_seconds: float) -> bool: ...

    async def wait_any(self, timeout_seconds: float) -> bool: ...


class InProcessVoiceTaskCompletionNotifier:
    """Low-latency process-local hint; PostgreSQL notification state remains truth."""

    def __init__(self) -> None:
        self._owners: dict[str, asyncio.Event] = {}
        self._any = asyncio.Event()

    async def wake(self, owner_person_id: str) -> None:
        owner = owner_person_id.strip()
        if not owner:
            raise ValueError("Delegated task notification owner must not be empty")
        self._owners.setdefault(owner, asyncio.Event()).set()
        self._any.set()

    async def wait_owner(self, owner_person_id: str, timeout_seconds: float) -> bool:
        event = self._owners.setdefault(owner_person_id, asyncio.Event())
        signalled = await _wait_event(event, timeout_seconds)
        if signalled:
            event.clear()
        return signalled

    async def wait_any(self, timeout_seconds: float) -> bool:
        signalled = await _wait_event(self._any, timeout_seconds)
        if signalled:
            self._any.clear()
        return signalled


class VoiceTaskMailboxService(VoiceTaskMailbox):
    """Map durable Agent task rows into the narrow active-session mailbox."""

    def __init__(
        self,
        repository: DelegatedTaskRepository,
        completion_signal: VoiceTaskCompletionSignal,
        text_gateway: VoiceTaskTextGateway,
    ) -> None:
        self._repository = repository
        self._signal = completion_signal
        self._text = text_gateway

    async def wait(self, owner_person_id: str, timeout_seconds: float) -> bool:
        return await self._signal.wait_owner(owner_person_id, timeout_seconds)

    async def claim(
        self,
        session_id: UUID,
        limit: int,
    ) -> tuple[VoiceTaskNotification, ...]:
        return tuple(
            self.project(task)
            for task in await self._repository.claim_notifications(session_id, limit)
        )

    async def present_text(self, notices: tuple[VoiceTaskNotification, ...]) -> bool:
        if not notices:
            return True
        groups: dict[VoiceTextAddress, list[VoiceTaskNotification]] = defaultdict(list)
        for notice in notices:
            groups[notice.origin].append(notice)
        succeeded = True
        for address, grouped in groups.items():
            ids = tuple(item.task_id for item in grouped)
            try:
                await self._text.send(address, render_task_notifications(grouped))
            except asyncio.CancelledError:
                await asyncio.shield(self._repository.defer_notifications(ids))
                raise
            except Exception as exc:
                succeeded = False
                logger.warning(
                    "Delegated task text notification failed: route=%s tasks=%s error=%s",
                    opaque_ref(address.area_id, address.channel_id),
                    len(ids),
                    exception_kind(exc),
                )
                await self._repository.defer_notifications(ids)
            else:
                await asyncio.shield(self._repository.mark_presented(ids))
                logger.info(
                    "Delegated task text notification presented: route=%s tasks=%s",
                    opaque_ref(address.area_id, address.channel_id),
                    len(ids),
                )
        return succeeded

    async def mark_presented(self, task_ids: tuple[UUID, ...]) -> None:
        await self._repository.mark_presented(task_ids)

    async def defer(self, task_ids: tuple[UUID, ...]) -> None:
        await self._repository.defer_notifications(task_ids)

    @staticmethod
    def project(task: DelegatedAgentTask) -> VoiceTaskNotification:
        return VoiceTaskNotification(
            task.id,
            task.alias,
            VoiceTaskNotificationStatus(task.status.value),
            task.objective,
            task.result_summary,
            task.error_message,
            VoiceTextAddress(task.area_id, task.text_channel_id),
        )


class DelegatedTaskTextFallbackReconciler:
    """Deliver terminal tasks whose originating voice session has already ended."""

    def __init__(
        self,
        repository: DelegatedTaskRepository,
        signal: VoiceTaskCompletionSignal,
        mailbox: VoiceTaskMailboxService,
        *,
        poll_seconds: float = 2.0,
        claim_limit: int = 30,
        group_limit: int = 3,
    ) -> None:
        if poll_seconds <= 0 or claim_limit <= 0 or group_limit <= 0:
            raise ValueError("Text fallback bounds must be positive")
        self._repository = repository
        self._signal = signal
        self._mailbox = mailbox
        self._poll_seconds = poll_seconds
        self._claim_limit = claim_limit
        self._group_limit = group_limit
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        recovered = await self._repository.recover_claimed_notifications()
        if recovered:
            logger.warning("Recovered claimed delegated notifications: count=%s", recovered)
        self._task = asyncio.create_task(
            self._run(),
            name="delegated-task-text-fallback",
        )
        logger.info("Delegated task text fallback started")

    async def aclose(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        try:
            await self._reconcile()
        except Exception as exc:
            logger.warning(
                "Final delegated task text reconciliation failed: error=%s",
                exception_kind(exc),
            )
        logger.info("Delegated task text fallback stopped")

    async def _run(self) -> None:
        while True:
            try:
                await self._reconcile()
                await self._signal.wait_any(self._poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Delegated task text reconciliation failed: error=%s",
                    exception_kind(exc),
                    exc_info=True,
                )
                await asyncio.sleep(self._poll_seconds)

    async def _reconcile(self) -> None:
        tasks = await self._repository.claim_text_notifications(self._claim_limit)
        if not tasks:
            return
        grouped: dict[VoiceTextAddress, list[DelegatedAgentTask]] = defaultdict(list)
        for task in tasks:
            grouped[VoiceTextAddress(task.area_id, task.text_channel_id)].append(task)
        for grouped_tasks in grouped.values():
            for batch in _chunks(grouped_tasks, self._group_limit):
                await self._mailbox.present_text(
                    tuple(self._mailbox.project(task) for task in batch)
                )


def render_task_notifications(notices: Iterable[VoiceTaskNotification]) -> str:
    lines: list[str] = []
    for notice in notices:
        objective = _line(notice.objective, 72)
        if notice.status is VoiceTaskNotificationStatus.SUCCEEDED:
            icon = "✅"
            detail = _line(notice.summary, 180) or "任务已完成"
        elif notice.status is VoiceTaskNotificationStatus.CANCELLED:
            icon = "⏹️"
            detail = "已取消"
        elif notice.status is VoiceTaskNotificationStatus.INTERRUPTED:
            icon = "⚠️"
            detail = _line(notice.error_message, 180) or "执行被中断"
        else:
            icon = "⚠️"
            detail = _line(notice.error_message, 180) or "执行失败"
        lines.append(f"{icon} **后台任务 {notice.alias}** {objective} · {detail}")
    return "\n".join(lines)


async def _wait_event(event: asyncio.Event, timeout_seconds: float) -> bool:
    if timeout_seconds <= 0:
        raise ValueError("Notification wait timeout must be positive")
    try:
        async with asyncio.timeout(timeout_seconds):
            await event.wait()
    except TimeoutError:
        return False
    return True


def _chunks[T](values: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _line(value: str, limit: int) -> str:
    normalized = _WHITESPACE.sub(" ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"

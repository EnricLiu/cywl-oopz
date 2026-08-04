"""Single-process scheduler for durable delegated Agent tasks."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from cywl_oopz.core.observability import exception_kind, opaque_ref

from .models import DelegatedAgentTask, DelegatedTaskLane, DelegatedTaskStatus
from .ports import DelegatedTaskRepository, DelegatedTaskWakeup

logger = logging.getLogger(__name__)


class DelegatedTaskRunner(Protocol):
    async def run(self, task: DelegatedAgentTask, worker_id: str) -> None: ...


@dataclass(slots=True)
class _ActiveTask:
    envelope: DelegatedAgentTask
    operation: asyncio.Task[None]
    cancel_reason: str = ""


class DelegatedTaskScheduler:
    """Claim short transactions and own every background task until shutdown."""

    def __init__(
        self,
        repository: DelegatedTaskRepository,
        wakeup: DelegatedTaskWakeup,
        runner: DelegatedTaskRunner,
        *,
        read_concurrency: int = 2,
        per_user_concurrency: int = 1,
        reconcile_seconds: float = 2.0,
        worker_id: str | None = None,
    ) -> None:
        if read_concurrency <= 0 or per_user_concurrency <= 0:
            raise ValueError("Delegated task concurrency must be positive")
        if per_user_concurrency > read_concurrency:
            raise ValueError("Per-user task concurrency cannot exceed read concurrency")
        if reconcile_seconds <= 0:
            raise ValueError("Delegated task reconciliation interval must be positive")
        self._repository = repository
        self._wakeup = wakeup
        self._runner = runner
        self._read_concurrency = read_concurrency
        self._per_user_concurrency = per_user_concurrency
        self._reconcile_seconds = reconcile_seconds
        self._worker_id = worker_id or f"cywl-{uuid4().hex[:16]}"
        self._active: dict[UUID, _ActiveTask] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._closing = False
        self._started = False

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def start(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._closing = False
        recovery = await self._repository.recover_stale(datetime.now(UTC))
        if recovery.task_ids:
            logger.warning(
                "Recovered stale delegated tasks: requeued=%s cancelled=%s interrupted=%s",
                recovery.requeued,
                recovery.cancelled,
                recovery.interrupted,
            )
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name="delegated-task-scheduler",
        )
        self._started = True
        logger.info(
            "Delegated task scheduler started: worker=%s read_concurrency=%s "
            "per_user_concurrency=%s",
            self._worker_id,
            self._read_concurrency,
            self._per_user_concurrency,
        )

    async def aclose(self) -> None:
        if not self._started:
            return
        self._closing = True
        loop_task = self._loop_task
        self._loop_task = None
        if loop_task is not None:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

        active = tuple(self._active.values())
        for item in active:
            item.cancel_reason = "shutdown"
            item.operation.cancel()
        if active:
            await asyncio.gather(*(item.operation for item in active), return_exceptions=True)
        self._active.clear()

        try:
            recovery = await self._repository.recover_stale(datetime.now(UTC))
        except Exception as exc:
            logger.warning(
                "Could not release delegated tasks during shutdown: error=%s",
                exception_kind(exc),
            )
        else:
            if recovery.task_ids:
                logger.info(
                    "Delegated tasks released during shutdown: requeued=%s cancelled=%s "
                    "interrupted=%s",
                    recovery.requeued,
                    recovery.cancelled,
                    recovery.interrupted,
                )
        self._started = False
        logger.info("Delegated task scheduler stopped: worker=%s", self._worker_id)

    async def _run_loop(self) -> None:
        while not self._closing:
            try:
                await self._fill_capacity()
                changed_ids = await self._wakeup.wait(self._reconcile_seconds)
                await self._cancel_requested(changed_ids)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Delegated task scheduler reconciliation failed: error=%s",
                    exception_kind(exc),
                    exc_info=True,
                )
                await asyncio.sleep(self._reconcile_seconds)

    async def _fill_capacity(self) -> None:
        read_active = [
            item
            for item in self._active.values()
            if item.envelope.lane is DelegatedTaskLane.READ_PARALLEL
        ]
        owner_counts = Counter(item.envelope.owner_person_id for item in read_active)
        while len(read_active) < self._read_concurrency:
            excluded = frozenset(
                owner
                for owner, count in owner_counts.items()
                if count >= self._per_user_concurrency
            )
            claimed = await self._repository.claim_next(
                self._worker_id,
                frozenset({DelegatedTaskLane.READ_PARALLEL}),
                excluded_owner_person_ids=excluded,
            )
            if claimed is None:
                break
            self._start_claimed(claimed)
            read_active.append(self._active[claimed.id])
            owner_counts[claimed.owner_person_id] += 1

        mutation_active = any(
            item.envelope.lane is DelegatedTaskLane.MUTATION_SERIAL
            for item in self._active.values()
        )
        if not mutation_active:
            claimed = await self._repository.claim_next(
                self._worker_id,
                frozenset({DelegatedTaskLane.MUTATION_SERIAL}),
            )
            if claimed is not None:
                self._start_claimed(claimed)

    def _start_claimed(self, task: DelegatedAgentTask) -> None:
        operation = asyncio.create_task(
            self._run_claimed(task),
            name=f"delegated-agent-task:{task.id}",
        )
        self._active[task.id] = _ActiveTask(task, operation)
        logger.debug(
            "Delegated task claimed: task=%s lane=%s active=%s",
            opaque_ref(str(task.id)),
            task.lane.value,
            len(self._active),
        )

    async def _run_claimed(self, task: DelegatedAgentTask) -> None:
        try:
            await self._runner.run(task, self._worker_id)
        except asyncio.CancelledError:
            active = self._active.get(task.id)
            current = await self._repository.get(task.id)
            if (
                active is not None
                and active.cancel_reason == "user"
                and current is not None
                and current.status is DelegatedTaskStatus.CANCEL_REQUESTED
            ):
                await self._repository.mark_cancelled(task.id, self._worker_id)
            raise
        except Exception as exc:
            logger.error(
                "Delegated task runner escaped unexpectedly: task=%s error=%s",
                opaque_ref(str(task.id)),
                exception_kind(exc),
                exc_info=True,
            )
        finally:
            self._active.pop(task.id, None)
            await self._wakeup.wake(task.id)

    async def _cancel_requested(self, changed_ids: tuple[UUID, ...]) -> None:
        candidates = (
            tuple(task_id for task_id in changed_ids if task_id in self._active)
            if changed_ids
            else tuple(self._active)
        )
        for task_id in candidates:
            active = self._active.get(task_id)
            if active is None or active.operation.done():
                continue
            current = await self._repository.get(task_id)
            if current is None or current.status is not DelegatedTaskStatus.CANCEL_REQUESTED:
                continue
            active.cancel_reason = "user"
            active.operation.cancel()
            logger.info(
                "Cancelling active delegated task: task=%s",
                opaque_ref(str(task_id)),
            )

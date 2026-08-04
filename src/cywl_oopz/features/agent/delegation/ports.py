"""Ports for durable delegated task storage and scheduler notification."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from .models import (
    CancelOutcome,
    DelegatedAgentTask,
    DelegatedTaskLane,
    DelegatedTaskPolicy,
    DelegatedTaskSubmission,
    RecoverySummary,
    TaskListQuery,
    TaskRef,
)


class DelegatedTaskRepository(Protocol):
    async def resolve_submission_policy(
        self,
        session_id: UUID,
        owner_person_id: str,
    ) -> DelegatedTaskPolicy: ...

    async def submit(self, request: DelegatedTaskSubmission) -> DelegatedAgentTask: ...

    async def get_for_owner(
        self,
        task_ref: TaskRef,
        owner_person_id: str,
    ) -> DelegatedAgentTask | None: ...

    async def list_for_owner(
        self,
        owner_person_id: str,
        query: TaskListQuery,
    ) -> tuple[DelegatedAgentTask, ...]: ...

    async def request_cancel(
        self,
        task_id: UUID,
        owner_person_id: str,
    ) -> CancelOutcome: ...

    async def claim_next(
        self,
        worker_id: str,
        lanes: frozenset[DelegatedTaskLane],
    ) -> DelegatedAgentTask | None: ...

    async def heartbeat(self, task_id: UUID, worker_id: str) -> bool: ...

    async def update_progress(
        self,
        task_id: UUID,
        worker_id: str,
        stage: str,
        summary: str,
    ) -> bool: ...

    async def mark_waiting_retry(
        self,
        task_id: UUID,
        worker_id: str,
        next_attempt_at: datetime,
        error_code: str,
        error_message: str = "",
    ) -> bool: ...

    async def complete(
        self,
        task_id: UUID,
        worker_id: str,
        result_summary: str,
        result_text: str,
        *,
        agent_thread_id: UUID | None = None,
        agent_run_id: UUID | None = None,
    ) -> bool: ...

    async def fail(
        self,
        task_id: UUID,
        worker_id: str,
        error_code: str,
        error_message: str = "",
    ) -> bool: ...

    async def mark_cancelled(self, task_id: UUID, worker_id: str) -> bool: ...

    async def claim_notifications(
        self,
        session_id: UUID,
        limit: int,
    ) -> tuple[DelegatedAgentTask, ...]: ...

    async def mark_presented(self, task_ids: tuple[UUID, ...]) -> None: ...

    async def recover_stale(self, now: datetime) -> RecoverySummary: ...


class DelegatedTaskWakeup(Protocol):
    async def wake(self, task_id: UUID) -> None: ...

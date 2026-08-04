"""Short-transaction PostgreSQL repository for delegated Agent tasks."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    DelegatedAgentTaskRecord,
    LlmModelRecord,
    LlmProviderRecord,
    UserLlmPreferenceRecord,
    VoiceChannelSettingsRecord,
    VoiceSessionRecord,
    VoiceUserPreferenceRecord,
)

from .models import (
    CancelOutcome,
    DelegatedAgentTask,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskPolicy,
    DelegatedTaskStatus,
    DelegatedTaskSubmission,
    RecoverySummary,
    TaskListQuery,
    TaskRef,
)


class SqlAlchemyDelegatedTaskRepository:
    """Keep task truth in PostgreSQL without retaining an in-process snapshot."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def resolve_submission_policy(
        self,
        session_id: UUID,
        owner_person_id: str,
    ) -> DelegatedTaskPolicy:
        try:
            async with self._sessions() as session:
                voice = await session.scalar(
                    select(VoiceSessionRecord).where(
                        VoiceSessionRecord.id == session_id,
                        VoiceSessionRecord.owner_person_id == owner_person_id,
                    )
                )
                if voice is None:
                    raise DatabaseError("Voice session is not available for this owner")
                channel = await session.get(
                    VoiceChannelSettingsRecord,
                    (voice.area_id, voice.voice_channel_id),
                )
                if channel is None or not channel.enabled:
                    raise DatabaseError("Voice delegated tasks are disabled in this channel")
                preference = await session.get(VoiceUserPreferenceRecord, owner_person_id)
                model_id = preference.delegated_agent_model_id if preference is not None else None
                if model_id is None:
                    model_id = await session.scalar(
                        select(UserLlmPreferenceRecord.preferred_model_id).where(
                            UserLlmPreferenceRecord.person_id == owner_person_id
                        )
                    )
                model_statement = (
                    select(LlmModelRecord.id)
                    .join(LlmProviderRecord, LlmProviderRecord.id == LlmModelRecord.provider_id)
                    .where(
                        LlmModelRecord.enabled.is_(True),
                        LlmProviderRecord.enabled.is_(True),
                    )
                )
                if model_id is None:
                    model_statement = model_statement.where(
                        LlmModelRecord.is_application_default.is_(True)
                    )
                else:
                    model_statement = model_statement.where(LlmModelRecord.id == model_id)
                resolved_model_id = await session.scalar(model_statement)
                if resolved_model_id is None:
                    raise DatabaseError("No enabled delegated Agent model is available")
                return DelegatedTaskPolicy(
                    profile=channel.delegated_task_profile,
                    agent_model_id=resolved_model_id,
                )
        except SQLAlchemyError as exc:
            raise _database_error("resolve delegated task policy", exc) from exc

    async def submit(self, request: DelegatedTaskSubmission) -> DelegatedAgentTask:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    existing = await self._submission_record(session, request)
                    if existing is not None:
                        return _task(existing)
                    voice = await session.scalar(
                        select(VoiceSessionRecord)
                        .where(VoiceSessionRecord.id == request.origin_voice_session_id)
                        .with_for_update()
                    )
                    if voice is None or voice.owner_person_id != request.owner_person_id:
                        raise DatabaseError("Voice session is not available for this owner")
                    if (
                        voice.area_id != request.area_id
                        or voice.voice_channel_id != request.voice_channel_id
                        or voice.text_channel_id != request.text_channel_id
                    ):
                        raise DatabaseError("Delegated task route does not match its voice session")
                    existing = await self._submission_record(session, request)
                    if existing is not None:
                        return _task(existing)
                    sequence = (
                        await session.scalar(
                            select(func.max(DelegatedAgentTaskRecord.session_sequence)).where(
                                DelegatedAgentTaskRecord.origin_voice_session_id
                                == request.origin_voice_session_id
                            )
                        )
                        or 0
                    ) + 1
                    record = DelegatedAgentTaskRecord(
                        owner_person_id=request.owner_person_id,
                        area_id=request.area_id,
                        text_channel_id=request.text_channel_id,
                        voice_channel_id=request.voice_channel_id,
                        origin_voice_session_id=request.origin_voice_session_id,
                        session_sequence=sequence,
                        provider_call_id=request.provider_call_id,
                        objective=request.objective,
                        result_style=request.result_style,
                        lane=request.lane,
                        conflict_key=request.conflict_key,
                        agent_model_id=request.agent_model_id,
                        allowed_tool_names=list(request.allowed_tool_names),
                    )
                    session.add(record)
                    await session.flush()
                    return _task(record)
        except SQLAlchemyError as exc:
            raise _database_error("submit delegated task", exc) from exc

    async def get_for_owner(
        self,
        task_ref: TaskRef,
        owner_person_id: str,
    ) -> DelegatedAgentTask | None:
        statement = select(DelegatedAgentTaskRecord).where(
            DelegatedAgentTaskRecord.owner_person_id == owner_person_id
        )
        if task_ref.task_id is not None:
            statement = statement.where(DelegatedAgentTaskRecord.id == task_ref.task_id)
        elif task_ref.origin_voice_session_id is not None and task_ref.session_sequence is not None:
            statement = statement.where(
                DelegatedAgentTaskRecord.origin_voice_session_id
                == task_ref.origin_voice_session_id,
                DelegatedAgentTaskRecord.session_sequence == task_ref.session_sequence,
            )
        else:
            raise ValueError("Task reference is incomplete")
        try:
            async with self._sessions() as session:
                record = await session.scalar(statement)
                return _task(record) if record is not None else None
        except SQLAlchemyError as exc:
            raise _database_error("read delegated task", exc) from exc

    async def get(self, task_id: UUID) -> DelegatedAgentTask | None:
        try:
            async with self._sessions() as session:
                record = await session.get(DelegatedAgentTaskRecord, task_id)
                return _task(record) if record is not None else None
        except SQLAlchemyError as exc:
            raise _database_error("read delegated task for scheduler", exc) from exc

    async def list_for_owner(
        self,
        owner_person_id: str,
        query: TaskListQuery,
    ) -> tuple[DelegatedAgentTask, ...]:
        statement = select(DelegatedAgentTaskRecord).where(
            DelegatedAgentTaskRecord.owner_person_id == owner_person_id
        )
        if query.status is not None:
            statement = statement.where(DelegatedAgentTaskRecord.status == query.status)
        if query.origin_voice_session_id is not None:
            statement = statement.where(
                DelegatedAgentTaskRecord.origin_voice_session_id == query.origin_voice_session_id
            )
        statement = statement.order_by(DelegatedAgentTaskRecord.created_at.desc()).limit(
            query.limit
        )
        try:
            async with self._sessions() as session:
                records = (await session.scalars(statement)).all()
                return tuple(_task(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("list delegated tasks", exc) from exc

    async def request_cancel(
        self,
        task_id: UUID,
        owner_person_id: str,
    ) -> CancelOutcome:
        now = datetime.now(UTC)
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(
                        select(DelegatedAgentTaskRecord)
                        .where(
                            DelegatedAgentTaskRecord.id == task_id,
                            DelegatedAgentTaskRecord.owner_person_id == owner_person_id,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        return CancelOutcome(None, False)
                    if record.status.terminal:
                        return CancelOutcome(_task(record), False, already_terminal=True)
                    if record.status in {
                        DelegatedTaskStatus.QUEUED,
                        DelegatedTaskStatus.WAITING_RETRY,
                    }:
                        record.status = DelegatedTaskStatus.CANCELLED
                        record.finished_at = now
                        record.notification_state = DelegatedTaskNotificationState.PENDING
                    else:
                        record.status = DelegatedTaskStatus.CANCEL_REQUESTED
                    if record.cancel_requested_at is None:
                        record.cancel_requested_at = now
                    await session.flush()
                    await session.refresh(record)
                    return CancelOutcome(_task(record), True)
        except SQLAlchemyError as exc:
            raise _database_error("cancel delegated task", exc) from exc

    async def claim_next(
        self,
        worker_id: str,
        lanes: frozenset[DelegatedTaskLane],
        *,
        excluded_owner_person_ids: frozenset[str] = frozenset(),
    ) -> DelegatedAgentTask | None:
        normalized_worker = worker_id.strip()
        if not normalized_worker or len(normalized_worker) > 128 or not lanes:
            raise ValueError("Worker identifier and lanes are required")
        now = datetime.now(UTC)
        try:
            async with self._sessions() as session:
                async with session.begin():
                    statement = (
                        select(DelegatedAgentTaskRecord)
                        .where(
                            DelegatedAgentTaskRecord.status.in_(
                                {
                                    DelegatedTaskStatus.QUEUED,
                                    DelegatedTaskStatus.WAITING_RETRY,
                                }
                            ),
                            DelegatedAgentTaskRecord.lane.in_(lanes),
                            or_(
                                DelegatedAgentTaskRecord.next_attempt_at.is_(None),
                                DelegatedAgentTaskRecord.next_attempt_at <= now,
                            ),
                        )
                        .order_by(DelegatedAgentTaskRecord.created_at)
                        .with_for_update(skip_locked=True)
                        .limit(1)
                    )
                    if excluded_owner_person_ids:
                        statement = statement.where(
                            DelegatedAgentTaskRecord.owner_person_id.not_in(
                                excluded_owner_person_ids
                            )
                        )
                    record = await session.scalar(statement)
                    if record is None:
                        return None
                    record.status = DelegatedTaskStatus.RUNNING
                    record.worker_id = normalized_worker
                    record.started_at = record.started_at or now
                    record.heartbeat_at = now
                    record.next_attempt_at = None
                    await session.flush()
                    await session.refresh(record)
                    return _task(record)
        except SQLAlchemyError as exc:
            raise _database_error("claim delegated task", exc) from exc

    async def heartbeat(self, task_id: UUID, worker_id: str) -> bool:
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status.in_(
                    {DelegatedTaskStatus.RUNNING, DelegatedTaskStatus.CANCEL_REQUESTED}
                ),
            )
            .values(heartbeat_at=datetime.now(UTC)),
            "heartbeat delegated task",
        )

    async def update_progress(
        self,
        task_id: UUID,
        worker_id: str,
        stage: str,
        summary: str,
    ) -> bool:
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status == DelegatedTaskStatus.RUNNING,
            )
            .values(
                progress_stage=_bounded(stage, 64),
                progress_summary=_bounded(summary, 512),
                heartbeat_at=datetime.now(UTC),
            ),
            "update delegated task progress",
        )

    async def mark_waiting_retry(
        self,
        task_id: UUID,
        worker_id: str,
        next_attempt_at: datetime,
        error_code: str,
        error_message: str = "",
    ) -> bool:
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status == DelegatedTaskStatus.RUNNING,
                DelegatedAgentTaskRecord.lane == DelegatedTaskLane.READ_PARALLEL,
            )
            .values(
                status=DelegatedTaskStatus.WAITING_RETRY,
                retry_count=DelegatedAgentTaskRecord.retry_count + 1,
                next_attempt_at=next_attempt_at,
                worker_id="",
                heartbeat_at=None,
                error_code=_bounded(error_code, 128),
                error_message=_bounded(error_message, 1000),
            ),
            "schedule delegated task retry",
        )

    async def complete(
        self,
        task_id: UUID,
        worker_id: str,
        result_summary: str,
        result_text: str,
        *,
        agent_thread_id: UUID | None = None,
        agent_run_id: UUID | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status == DelegatedTaskStatus.RUNNING,
            )
            .values(
                status=DelegatedTaskStatus.SUCCEEDED,
                result_summary=_bounded(result_summary, 1000),
                result_text=_bounded(result_text, 16000),
                agent_thread_id=agent_thread_id,
                agent_run_id=agent_run_id,
                finished_at=now,
                heartbeat_at=now,
                notification_state=DelegatedTaskNotificationState.PENDING,
            ),
            "complete delegated task",
        )

    async def fail(
        self,
        task_id: UUID,
        worker_id: str,
        error_code: str,
        error_message: str = "",
    ) -> bool:
        now = datetime.now(UTC)
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status == DelegatedTaskStatus.RUNNING,
            )
            .values(
                status=DelegatedTaskStatus.FAILED,
                error_code=_bounded(error_code, 128),
                error_message=_bounded(error_message, 1000),
                finished_at=now,
                heartbeat_at=now,
                notification_state=DelegatedTaskNotificationState.PENDING,
            ),
            "fail delegated task",
        )

    async def mark_cancelled(self, task_id: UUID, worker_id: str) -> bool:
        now = datetime.now(UTC)
        return await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id == task_id,
                DelegatedAgentTaskRecord.worker_id == worker_id,
                DelegatedAgentTaskRecord.status.in_(
                    {DelegatedTaskStatus.RUNNING, DelegatedTaskStatus.CANCEL_REQUESTED}
                ),
            )
            .values(
                status=DelegatedTaskStatus.CANCELLED,
                finished_at=now,
                heartbeat_at=now,
                notification_state=DelegatedTaskNotificationState.PENDING,
            ),
            "mark delegated task cancelled",
        )

    async def claim_notifications(
        self,
        session_id: UUID,
        limit: int,
    ) -> tuple[DelegatedAgentTask, ...]:
        if not 1 <= limit <= 20:
            raise ValueError("Notification claim limit must be between 1 and 20")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    records = (
                        await session.scalars(
                            select(DelegatedAgentTaskRecord)
                            .where(
                                DelegatedAgentTaskRecord.origin_voice_session_id == session_id,
                                DelegatedAgentTaskRecord.status.in_(
                                    {
                                        DelegatedTaskStatus.SUCCEEDED,
                                        DelegatedTaskStatus.FAILED,
                                        DelegatedTaskStatus.CANCELLED,
                                        DelegatedTaskStatus.INTERRUPTED,
                                    }
                                ),
                                DelegatedAgentTaskRecord.notification_state.in_(
                                    {
                                        DelegatedTaskNotificationState.PENDING,
                                        DelegatedTaskNotificationState.DEFERRED,
                                    }
                                ),
                            )
                            .order_by(DelegatedAgentTaskRecord.finished_at)
                            .with_for_update(skip_locked=True)
                            .limit(limit)
                        )
                    ).all()
                    for record in records:
                        record.notification_state = DelegatedTaskNotificationState.CLAIMED
                    await session.flush()
                    for record in records:
                        await session.refresh(record)
                    return tuple(_task(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("claim delegated task notifications", exc) from exc

    async def mark_presented(self, task_ids: tuple[UUID, ...]) -> None:
        if not task_ids:
            return
        now = datetime.now(UTC)
        await self._conditional_update(
            update(DelegatedAgentTaskRecord)
            .where(
                DelegatedAgentTaskRecord.id.in_(task_ids),
                DelegatedAgentTaskRecord.notification_state
                == DelegatedTaskNotificationState.CLAIMED,
            )
            .values(
                notification_state=DelegatedTaskNotificationState.PRESENTED,
                presented_at=now,
            ),
            "mark delegated task notifications presented",
            require_row=False,
        )

    async def recover_stale(self, now: datetime) -> RecoverySummary:
        requeued: list[UUID] = []
        cancelled: list[UUID] = []
        interrupted: list[UUID] = []
        try:
            async with self._sessions() as session:
                async with session.begin():
                    records = (
                        await session.scalars(
                            select(DelegatedAgentTaskRecord)
                            .where(
                                DelegatedAgentTaskRecord.status.in_(
                                    {
                                        DelegatedTaskStatus.RUNNING,
                                        DelegatedTaskStatus.CANCEL_REQUESTED,
                                    }
                                )
                            )
                            .with_for_update(skip_locked=True)
                        )
                    ).all()
                    for record in records:
                        record.worker_id = ""
                        record.heartbeat_at = None
                        if record.status is DelegatedTaskStatus.CANCEL_REQUESTED:
                            record.status = DelegatedTaskStatus.CANCELLED
                            record.finished_at = now
                            cancelled.append(record.id)
                        elif record.lane is DelegatedTaskLane.READ_PARALLEL:
                            record.status = DelegatedTaskStatus.QUEUED
                            record.retry_count += 1
                            record.next_attempt_at = now
                            requeued.append(record.id)
                        else:
                            record.status = DelegatedTaskStatus.INTERRUPTED
                            record.finished_at = now
                            record.error_code = "stale_mutation"
                            interrupted.append(record.id)
                        if record.status.terminal:
                            record.notification_state = DelegatedTaskNotificationState.PENDING
                    await session.flush()
            ids = tuple(requeued + cancelled + interrupted)
            return RecoverySummary(len(requeued), len(cancelled), len(interrupted), ids)
        except SQLAlchemyError as exc:
            raise _database_error("recover stale delegated tasks", exc) from exc

    async def _conditional_update(
        self,
        statement,
        operation: str,
        *,
        require_row: bool = True,
    ) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(statement)
                    changed = bool(result.rowcount)
                    if require_row and not changed:
                        return False
                    return changed
        except SQLAlchemyError as exc:
            raise _database_error(operation, exc) from exc

    @staticmethod
    async def _submission_record(
        session: AsyncSession,
        request: DelegatedTaskSubmission,
    ) -> DelegatedAgentTaskRecord | None:
        return await session.scalar(
            select(DelegatedAgentTaskRecord).where(
                DelegatedAgentTaskRecord.origin_voice_session_id == request.origin_voice_session_id,
                DelegatedAgentTaskRecord.provider_call_id == request.provider_call_id,
            )
        )


def _task(record: DelegatedAgentTaskRecord) -> DelegatedAgentTask:
    return DelegatedAgentTask(
        id=record.id,
        owner_person_id=record.owner_person_id,
        area_id=record.area_id,
        text_channel_id=record.text_channel_id,
        voice_channel_id=record.voice_channel_id,
        origin_voice_session_id=record.origin_voice_session_id,
        session_sequence=record.session_sequence,
        provider_call_id=record.provider_call_id,
        objective=record.objective,
        result_style=record.result_style,
        status=record.status,
        lane=record.lane,
        conflict_key=record.conflict_key,
        notification_state=record.notification_state,
        agent_model_id=record.agent_model_id,
        allowed_tool_names=tuple(record.allowed_tool_names),
        progress_stage=record.progress_stage,
        progress_summary=record.progress_summary,
        result_summary=record.result_summary,
        result_text=record.result_text,
        error_code=record.error_code,
        error_message=record.error_message,
        retry_count=record.retry_count,
        cancel_requested_at=record.cancel_requested_at,
        next_attempt_at=record.next_attempt_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
        agent_thread_id=record.agent_thread_id,
        agent_run_id=record.agent_run_id,
    )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    return DatabaseError(f"Failed to {operation}: {type(error).__name__}")


def _bounded(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]

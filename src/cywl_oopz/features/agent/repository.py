"""PostgreSQL repositories for the Agent provider catalog and selections."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.storage.models import (
    AgentMessageRecord,
    AgentRunRecord,
    AgentThreadRecord,
    AgentToolExecutionRecord,
    ChannelSettingsRecord,
    LlmModelRecord,
    LlmProviderRecord,
    UserLlmPreferenceRecord,
)

from .models import (
    AgentMessage,
    AgentRun,
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    AgentThread,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ModelSelectionCandidates,
    ProviderProtocol,
)
from .tools.models import (
    ToolExecution,
    ToolExecutionClaim,
    ToolExecutionStatus,
)

logger = logging.getLogger(__name__)


class SqlAlchemyProviderCatalogRepository:
    """Load provider/model configuration through short-lived ORM sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_providers(self) -> tuple[LlmProvider, ...]:
        """Load providers without logging credentials."""
        try:
            async with self._sessions() as session:
                records = (await session.scalars(select(LlmProviderRecord))).all()
                return tuple(self._provider_to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("load LLM providers", exc) from exc

    async def load_models(self) -> tuple[LlmModel, ...]:
        """Load configured models and their measured capability snapshots."""
        try:
            async with self._sessions() as session:
                records = (await session.scalars(select(LlmModelRecord))).all()
                return tuple(self._model_to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("load LLM models", exc) from exc

    async def upsert_provider_bundle(
        self,
        provider: LlmProvider,
        models: tuple[LlmModel, ...],
    ) -> None:
        """Upsert one provider and its supplied models in one transaction."""
        if any(model.provider_id != provider.id for model in models):
            raise ValueError("All models must belong to the provider being saved")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    provider_record = await session.get(LlmProviderRecord, provider.id)
                    if provider_record is None:
                        provider_record = LlmProviderRecord(id=provider.id)
                        session.add(provider_record)
                    self._update_provider_record(provider_record, provider)
                    await session.flush()

                    if any(model.is_provider_default for model in models):
                        await session.execute(
                            update(LlmModelRecord)
                            .where(LlmModelRecord.provider_id == provider.id)
                            .values(is_provider_default=False)
                        )
                    if any(model.is_application_default for model in models):
                        await session.execute(
                            update(LlmModelRecord).values(is_application_default=False)
                        )

                    records: list[tuple[LlmModelRecord, LlmModel]] = []
                    for model in models:
                        model_record = await session.get(LlmModelRecord, model.id)
                        if model_record is None:
                            model_record = LlmModelRecord(
                                id=model.id,
                                provider_id=provider.id,
                            )
                            session.add(model_record)
                        self._update_model_record(model_record, model, include_fallback=False)
                        records.append((model_record, model))
                    await session.flush()
                    for model_record, model in records:
                        model_record.fallback_model_id = model.fallback_model_id
        except SQLAlchemyError as exc:
            raise _database_error("save LLM provider configuration", exc) from exc

    @staticmethod
    def _provider_to_domain(record: LlmProviderRecord) -> LlmProvider:
        return LlmProvider(
            id=record.id,
            alias=record.alias,
            display_name=record.display_name,
            protocol=ProviderProtocol(record.protocol),
            base_url=record.base_url,
            api_key=record.api_key,
            user_selectable=record.user_selectable,
            enabled=record.enabled,
            config=_mapping(record.config),
        )

    @staticmethod
    def _model_to_domain(record: LlmModelRecord) -> LlmModel:
        raw_capabilities = record.capabilities if isinstance(record.capabilities, list) else []
        return LlmModel(
            id=record.id,
            provider_id=record.provider_id,
            alias=record.alias,
            remote_model_name=record.remote_model_name,
            display_name=record.display_name,
            enabled=record.enabled,
            is_provider_default=record.is_provider_default,
            is_application_default=record.is_application_default,
            capabilities=frozenset(ModelCapability(str(item)) for item in raw_capabilities),
            limits=_mapping(record.limits),
            fallback_model_id=record.fallback_model_id,
            pricing=_mapping(record.pricing),
        )

    @staticmethod
    def _update_provider_record(
        record: LlmProviderRecord,
        provider: LlmProvider,
    ) -> None:
        record.alias = provider.alias
        record.display_name = provider.display_name
        record.protocol = provider.protocol.value
        record.base_url = provider.base_url
        record.api_key = provider.api_key
        record.user_selectable = provider.user_selectable
        record.enabled = provider.enabled
        record.config = dict(provider.config)

    @staticmethod
    def _update_model_record(
        record: LlmModelRecord,
        model: LlmModel,
        *,
        include_fallback: bool,
    ) -> None:
        record.provider_id = model.provider_id
        record.alias = model.alias
        record.remote_model_name = model.remote_model_name
        record.display_name = model.display_name
        record.enabled = model.enabled
        record.is_provider_default = model.is_provider_default
        record.is_application_default = model.is_application_default
        record.capabilities = sorted(capability.value for capability in model.capabilities)
        record.limits = dict(model.limits)
        record.fallback_model_id = model.fallback_model_id if include_fallback else None
        record.pricing = dict(model.pricing)


class SqlAlchemyModelSelectionRepository:
    """Read all selection layers in one short database session."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_candidates(self, key: ConversationKey) -> ModelSelectionCandidates:
        """Load model IDs without evaluating enabled state or model capability."""
        try:
            async with self._sessions() as session:
                thread_model_id = await session.scalar(
                    select(AgentThreadRecord.selected_model_id).where(
                        AgentThreadRecord.scope == key.scope,
                        AgentThreadRecord.area_id == key.area_id,
                        AgentThreadRecord.channel_id == key.channel_id,
                        AgentThreadRecord.person_id == key.person_id,
                    )
                )
                user_model_id = await session.scalar(
                    select(UserLlmPreferenceRecord.preferred_model_id).where(
                        UserLlmPreferenceRecord.person_id == key.person_id
                    )
                )
                channel_model_id = None
                if key.scope == "channel":
                    channel_model_id = await session.scalar(
                        select(ChannelSettingsRecord.default_model_id).where(
                            ChannelSettingsRecord.area_id == key.area_id,
                            ChannelSettingsRecord.channel_id == key.channel_id,
                        )
                    )
                application_model_id = await session.scalar(
                    select(LlmModelRecord.id).where(LlmModelRecord.is_application_default.is_(True))
                )
                return ModelSelectionCandidates(
                    thread_model_id=thread_model_id,
                    user_model_id=user_model_id,
                    channel_model_id=channel_model_id,
                    application_model_id=application_model_id,
                )
        except SQLAlchemyError as exc:
            raise _database_error("load LLM model selection", exc) from exc

    async def set_user_model(self, person_id: str, model_id: UUID) -> None:
        """Insert or update one user's default model."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.get(UserLlmPreferenceRecord, person_id)
                    if record is None:
                        session.add(
                            UserLlmPreferenceRecord(
                                person_id=person_id,
                                preferred_model_id=model_id,
                            )
                        )
                    else:
                        record.preferred_model_id = model_id
        except SQLAlchemyError as exc:
            raise _database_error("save user LLM preference", exc) from exc


class SqlAlchemyAgentThreadRepository:
    """Persist Agent thread metadata without keeping ORM sessions across model I/O."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, key: ConversationKey) -> AgentThread | None:
        """Load one thread by its current privacy scope."""
        try:
            async with self._sessions() as session:
                record = await session.scalar(self._query(key))
                return None if record is None else self._to_domain(record)
        except SQLAlchemyError as exc:
            raise _database_error("load Agent thread", exc) from exc

    async def add(self, thread: AgentThread) -> None:
        """Create a thread in one short transaction."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(
                        AgentThreadRecord(
                            id=thread.id,
                            scope=thread.key.scope,
                            area_id=thread.key.area_id,
                            channel_id=thread.key.channel_id,
                            person_id=thread.key.person_id,
                            selected_model_id=thread.selected_model_id,
                            expires_at=thread.expires_at,
                            summary=thread.summary,
                            summary_through_sequence=thread.summary_through_sequence,
                            summary_version=thread.summary_version,
                            version=thread.version,
                        )
                    )
        except SQLAlchemyError as exc:
            raise _database_error("create Agent thread", exc) from exc

    async def set_selected_model(self, key: ConversationKey, model_id: UUID) -> None:
        """Pin an existing thread without changing its history."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(AgentThreadRecord)
                        .where(
                            AgentThreadRecord.scope == key.scope,
                            AgentThreadRecord.area_id == key.area_id,
                            AgentThreadRecord.channel_id == key.channel_id,
                            AgentThreadRecord.person_id == key.person_id,
                        )
                        .values(
                            selected_model_id=model_id,
                            version=AgentThreadRecord.version + 1,
                        )
                    )
                    if result.rowcount != 1:
                        raise DatabaseError("Agent thread does not exist")
        except SQLAlchemyError as exc:
            raise _database_error("select Agent thread model", exc) from exc

    async def refresh_expiry(self, thread_id: UUID, expires_at: datetime) -> None:
        """Extend a thread TTL after a completed interaction."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    await session.execute(
                        update(AgentThreadRecord)
                        .where(AgentThreadRecord.id == thread_id)
                        .values(expires_at=expires_at)
                    )
        except SQLAlchemyError as exc:
            raise _database_error("refresh Agent thread", exc) from exc

    async def save_summary(
        self,
        thread_id: UUID,
        summary: str,
        through_sequence: int,
        *,
        expected_version: int,
    ) -> bool:
        """Replace derived summary only if the thread version is unchanged."""
        if through_sequence <= 0 or not summary.strip():
            raise ValueError("Agent summary and sequence must not be empty")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(AgentThreadRecord)
                        .where(
                            AgentThreadRecord.id == thread_id,
                            AgentThreadRecord.version == expected_version,
                            AgentThreadRecord.summary_through_sequence < through_sequence,
                        )
                        .values(
                            summary=summary.strip(),
                            summary_through_sequence=through_sequence,
                            summary_version=AgentThreadRecord.summary_version + 1,
                            version=AgentThreadRecord.version + 1,
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("save Agent thread summary", exc) from exc

    async def delete(self, key: ConversationKey) -> None:
        """Delete one thread and let database cascades remove runtime records."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(self._query(key))
                    if record is not None:
                        await session.delete(record)
        except SQLAlchemyError as exc:
            raise _database_error("delete Agent thread", exc) from exc

    @staticmethod
    def _query(key: ConversationKey):
        return select(AgentThreadRecord).where(
            AgentThreadRecord.scope == key.scope,
            AgentThreadRecord.area_id == key.area_id,
            AgentThreadRecord.channel_id == key.channel_id,
            AgentThreadRecord.person_id == key.person_id,
        )

    @staticmethod
    def _to_domain(record: AgentThreadRecord) -> AgentThread:
        return AgentThread(
            id=record.id,
            key=ConversationKey(
                record.scope,
                record.area_id,
                record.channel_id,
                record.person_id,
            ),
            selected_model_id=record.selected_model_id,
            expires_at=record.expires_at,
            summary=record.summary,
            summary_through_sequence=record.summary_through_sequence,
            summary_version=record.summary_version,
            version=record.version,
        )


class SqlAlchemyAgentRunRepository:
    """Persist run lifecycle changes as independent short transactions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, run: AgentRun) -> None:
        """Persist a running record before any model or tool network I/O."""
        if (
            run.state.status is not AgentRunStatus.RUNNING
            or run.state.started_at is None
            or run.heartbeat_at is None
        ):
            raise ValueError("Only a started Agent run can be persisted")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(
                        AgentRunRecord(
                            id=run.id,
                            thread_id=run.thread_id,
                            status=run.state.status,
                            stop_reason=None,
                            provider_id=run.provider_id,
                            model_id=run.model_id,
                            selection_source=run.selection_source,
                            limits=asdict(run.limits),
                            usage=dict(run.usage),
                            error_code=run.error_code,
                            started_at=run.state.started_at,
                            heartbeat_at=run.heartbeat_at,
                        )
                    )
        except SQLAlchemyError as exc:
            raise _database_error("create Agent run", exc) from exc

    async def finish(
        self,
        state: AgentRunState,
        *,
        usage: dict[str, object],
        error_code: str = "",
    ) -> None:
        """Move an existing running record to a terminal state once."""
        if state.status in {AgentRunStatus.PENDING, AgentRunStatus.RUNNING}:
            raise ValueError("Agent run finish requires a terminal state")
        if state.finished_at is None or state.stop_reason is None:
            raise ValueError("Terminal Agent run state is incomplete")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(AgentRunRecord)
                        .where(
                            AgentRunRecord.id == state.run_id,
                            AgentRunRecord.status == AgentRunStatus.RUNNING,
                        )
                        .values(
                            status=state.status,
                            stop_reason=state.stop_reason,
                            usage=usage,
                            error_code=error_code,
                            finished_at=state.finished_at,
                            heartbeat_at=state.finished_at,
                        )
                    )
                    if result.rowcount != 1:
                        raise DatabaseError("Agent run is missing or already finished")
        except SQLAlchemyError as exc:
            raise _database_error("finish Agent run", exc) from exc

    async def heartbeat(self, run_id: UUID, now: datetime) -> bool:
        """Refresh one running record without extending any conversation lock."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(AgentRunRecord)
                        .where(
                            AgentRunRecord.id == run_id,
                            AgentRunRecord.status == AgentRunStatus.RUNNING,
                        )
                        .values(heartbeat_at=now)
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("heartbeat Agent run", exc) from exc

    async def abandon_stale(self, before: datetime, now: datetime) -> int:
        """Mark runs with an expired heartbeat abandoned after process restart."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(AgentRunRecord)
                        .where(
                            AgentRunRecord.status == AgentRunStatus.RUNNING,
                            AgentRunRecord.heartbeat_at < before,
                        )
                        .values(
                            status=AgentRunStatus.ABANDONED,
                            stop_reason=AgentStopReason.STALE_RUN_ABANDONED,
                            finished_at=now,
                            heartbeat_at=now,
                        )
                    )
                    return result.rowcount
        except SQLAlchemyError as exc:
            raise _database_error("abandon stale Agent runs", exc) from exc


class SqlAlchemyAgentMessageRepository:
    """Persist ordered messages while locking only the owning thread row."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(
        self,
        thread_id: UUID,
        *,
        limit: int,
        after_sequence: int = 0,
    ) -> tuple[AgentMessage, ...]:
        """Load a bounded suffix and return it in chronological order."""
        try:
            async with self._sessions() as session:
                records = (
                    await session.scalars(
                        select(AgentMessageRecord)
                        .outerjoin(
                            AgentRunRecord,
                            AgentRunRecord.id == AgentMessageRecord.run_id,
                        )
                        .where(AgentMessageRecord.thread_id == thread_id)
                        .where(AgentMessageRecord.sequence > after_sequence)
                        .where(
                            or_(
                                AgentMessageRecord.run_id.is_(None),
                                AgentRunRecord.status == AgentRunStatus.SUCCEEDED,
                            )
                        )
                        .order_by(AgentMessageRecord.sequence.desc())
                        .limit(limit)
                    )
                ).all()
                records.reverse()
                return tuple(self._to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("load Agent messages", exc) from exc

    async def load_after(
        self,
        thread_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> tuple[AgentMessage, ...]:
        """Load an oldest-first bounded window for deterministic summarization."""
        try:
            async with self._sessions() as session:
                records = (
                    await session.scalars(
                        select(AgentMessageRecord)
                        .outerjoin(
                            AgentRunRecord,
                            AgentRunRecord.id == AgentMessageRecord.run_id,
                        )
                        .where(
                            AgentMessageRecord.thread_id == thread_id,
                            AgentMessageRecord.sequence > after_sequence,
                        )
                        .where(
                            or_(
                                AgentMessageRecord.run_id.is_(None),
                                AgentRunRecord.status == AgentRunStatus.SUCCEEDED,
                            )
                        )
                        .order_by(AgentMessageRecord.sequence)
                        .limit(limit)
                    )
                ).all()
                return tuple(self._to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("load Agent summary messages", exc) from exc

    async def append(
        self,
        thread_id: UUID,
        run_id: UUID,
        messages: tuple[AgentMessage, ...],
    ) -> None:
        """Append one batch after serializing writers on the thread record."""
        if not messages:
            return
        try:
            async with self._sessions() as session:
                async with session.begin():
                    thread = await session.scalar(
                        select(AgentThreadRecord)
                        .where(AgentThreadRecord.id == thread_id)
                        .with_for_update()
                    )
                    if thread is None:
                        raise DatabaseError("Agent thread does not exist")
                    last_sequence = (
                        await session.scalar(
                            select(func.max(AgentMessageRecord.sequence)).where(
                                AgentMessageRecord.thread_id == thread_id
                            )
                        )
                        or 0
                    )
                    for offset, message in enumerate(messages, start=1):
                        session.add(
                            AgentMessageRecord(
                                thread_id=thread_id,
                                run_id=run_id,
                                sequence=last_sequence + offset,
                                role=message.role,
                                kind=message.kind,
                                content=dict(message.content),
                                input_tokens=message.input_tokens,
                                output_tokens=message.output_tokens,
                            )
                        )
        except SQLAlchemyError as exc:
            raise _database_error("append Agent messages", exc) from exc

    async def count(self, thread_id: UUID) -> int:
        """Count one thread without reading message content."""
        try:
            async with self._sessions() as session:
                return (
                    await session.scalar(
                        select(func.count(AgentMessageRecord.id)).where(
                            AgentMessageRecord.thread_id == thread_id,
                            or_(
                                AgentMessageRecord.run_id.is_(None),
                                AgentMessageRecord.run_id.in_(
                                    select(AgentRunRecord.id).where(
                                        AgentRunRecord.status == AgentRunStatus.SUCCEEDED
                                    )
                                ),
                            ),
                        )
                    )
                    or 0
                )
        except SQLAlchemyError as exc:
            raise _database_error("count Agent messages", exc) from exc

    @staticmethod
    def _to_domain(record: AgentMessageRecord) -> AgentMessage:
        return AgentMessage(
            role=record.role,
            kind=record.kind,
            content=_mapping(record.content),
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            sequence=record.sequence,
        )


class SqlAlchemyToolExecutionRepository:
    """Persist idempotent tool call claims and terminal results."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def claim(self, execution: ToolExecution) -> ToolExecutionClaim:
        """Insert one run/tool-call identity or return the existing record."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        postgresql_insert(AgentToolExecutionRecord)
                        .values(
                            id=execution.id,
                            run_id=execution.run_id,
                            tool_call_id=execution.call_id,
                            tool_name=execution.tool_name,
                            tool_version=execution.tool_version,
                            effect=execution.effect,
                            status=execution.status,
                            idempotency_key=execution.idempotency_key,
                            input_payload=dict(execution.input_payload),
                            output_payload=None,
                            error_code="",
                            started_at=execution.started_at,
                            finished_at=None,
                        )
                        .on_conflict_do_nothing()
                    )
                    if result.rowcount == 1:
                        return ToolExecutionClaim(execution, created=True)
                    record = await session.scalar(
                        select(AgentToolExecutionRecord).where(
                            AgentToolExecutionRecord.run_id == execution.run_id,
                            or_(
                                AgentToolExecutionRecord.tool_call_id == execution.call_id,
                                AgentToolExecutionRecord.idempotency_key
                                == execution.idempotency_key,
                            ),
                        )
                    )
                    if record is None:
                        raise DatabaseError("Tool execution claim was not persisted")
                    return ToolExecutionClaim(self._to_domain(record), created=False)
        except SQLAlchemyError as exc:
            raise _database_error("claim Agent tool execution", exc) from exc

    async def finish(
        self,
        run_id: UUID,
        call_id: str,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None,
        error_code: str,
    ) -> ToolExecution:
        """Finish a claimed call exactly once."""
        if status is ToolExecutionStatus.STARTED:
            raise ValueError("Tool execution finish requires a terminal status")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    finished_at = datetime.now(UTC)
                    result = await session.execute(
                        update(AgentToolExecutionRecord)
                        .where(
                            AgentToolExecutionRecord.run_id == run_id,
                            AgentToolExecutionRecord.tool_call_id == call_id,
                            AgentToolExecutionRecord.status == ToolExecutionStatus.STARTED,
                        )
                        .values(
                            status=status,
                            output_payload=output,
                            error_code=error_code,
                            finished_at=finished_at,
                        )
                    )
                    if result.rowcount != 1:
                        raise DatabaseError("Tool execution is missing or already finished")
                    record = await session.scalar(
                        select(AgentToolExecutionRecord).where(
                            AgentToolExecutionRecord.run_id == run_id,
                            AgentToolExecutionRecord.tool_call_id == call_id,
                        )
                    )
                    if record is None:
                        raise DatabaseError("Finished tool execution is missing")
                    return self._to_domain(record)
        except SQLAlchemyError as exc:
            raise _database_error("finish Agent tool execution", exc) from exc

    @staticmethod
    def _to_domain(record: AgentToolExecutionRecord) -> ToolExecution:
        return ToolExecution(
            id=record.id,
            run_id=record.run_id,
            call_id=record.tool_call_id,
            tool_name=record.tool_name,
            tool_version=record.tool_version,
            effect=record.effect,
            status=record.status,
            idempotency_key=record.idempotency_key,
            input_payload=_mapping(record.input_payload),
            output_payload=(
                _mapping(record.output_payload) if record.output_payload is not None else None
            ),
            error_code=record.error_code,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )


def _mapping(value: object) -> Mapping[str, Any]:
    """Copy a JSON object while rejecting non-object persisted values."""
    return value if isinstance(value, dict) else {}


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    """Report a static operation name without rendering SQL or record data."""
    logger.warning(
        "Agent persistence failed: operation=%s error=%s",
        operation,
        type(error).__name__,
    )
    return DatabaseError(f"Failed to {operation}")

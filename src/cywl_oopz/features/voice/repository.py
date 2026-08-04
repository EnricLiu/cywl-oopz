"""Short-lived PostgreSQL repositories for realtime voice configuration and history."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    VoiceChannelSettingsRecord,
    VoiceModelRecord,
    VoiceProviderRecord,
    VoiceSessionRecord,
    VoiceTurnRecord,
    VoiceUserPreferenceRecord,
)

from .errors import (
    VoiceChannelDisabledError,
    VoiceConfigurationUnavailableError,
    VoiceModelSelectionError,
    VoiceSpeakerSelectionError,
)
from .models import VoiceChannelKey, VoiceSessionDescriptor
from .settings import (
    PersistedVoiceSessionStatus,
    SelectableVoiceModel,
    VoiceChannelConfiguration,
    VoiceModelConfiguration,
    VoiceProviderConfiguration,
    VoiceStartConfiguration,
    VoiceTurnRole,
    VoiceUserSelection,
)


class SqlAlchemyVoiceConfigurationRepository:
    """Fresh-read Provider/model/channel/preference state for every operation."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def resolve_start_configuration(
        self,
        owner_person_id: str,
        channel: VoiceChannelKey,
    ) -> VoiceStartConfiguration:
        try:
            async with self._sessions() as session:
                channel_record = await session.get(
                    VoiceChannelSettingsRecord,
                    (channel.area_id, channel.channel_id),
                )
                if channel_record is None or not channel_record.enabled:
                    raise VoiceChannelDisabledError
                preference = await session.get(VoiceUserPreferenceRecord, owner_person_id)
                model_id = preference.preferred_model_id if preference is not None else None
                statement = self._enabled_model_statement()
                if model_id is None:
                    statement = statement.where(VoiceModelRecord.is_application_default.is_(True))
                else:
                    statement = statement.where(VoiceModelRecord.id == model_id)
                row = (await session.execute(statement)).one_or_none()
                if row is None:
                    raise VoiceConfigurationUnavailableError(
                        "No enabled selected or application-default voice model"
                    )
                model_record, provider_record = row
                selection = self._selection(preference)
                return VoiceStartConfiguration(
                    provider=self._provider(provider_record),
                    model=self._model(model_record),
                    channel=VoiceChannelConfiguration(
                        channel=channel,
                        delegated_task_profile=channel_record.delegated_task_profile,
                        idle_timeout_seconds=channel_record.idle_timeout_seconds,
                    ),
                    voice_id=selection.voice_id,
                    duplex_mode=selection.duplex_mode,
                    delegated_agent_model_id=selection.delegated_agent_model_id,
                )
        except SQLAlchemyError as exc:
            raise _database_error("resolve voice start configuration", exc) from exc

    async def list_selectable_models(
        self,
        owner_person_id: str,
    ) -> tuple[SelectableVoiceModel, ...]:
        try:
            async with self._sessions() as session:
                selected_id = await session.scalar(
                    select(VoiceUserPreferenceRecord.preferred_model_id).where(
                        VoiceUserPreferenceRecord.owner_person_id == owner_person_id
                    )
                )
                if selected_id is None:
                    selected_id = await session.scalar(
                        select(VoiceModelRecord.id)
                        .join(
                            VoiceProviderRecord,
                            VoiceProviderRecord.id == VoiceModelRecord.provider_id,
                        )
                        .where(
                            VoiceModelRecord.is_application_default.is_(True),
                            VoiceModelRecord.enabled.is_(True),
                            VoiceProviderRecord.enabled.is_(True),
                            VoiceProviderRecord.user_selectable.is_(True),
                        )
                    )
                rows = (
                    await session.execute(
                        self._enabled_model_statement()
                        .where(VoiceProviderRecord.user_selectable.is_(True))
                        .order_by(VoiceProviderRecord.alias, VoiceModelRecord.alias)
                    )
                ).all()
                return tuple(
                    SelectableVoiceModel(
                        id=model.id,
                        provider_alias=provider.alias,
                        model_alias=model.alias,
                        display_name=model.display_name,
                        mode=model.mode,
                        selected=model.id == selected_id,
                    )
                    for model, provider in rows
                )
        except SQLAlchemyError as exc:
            raise _database_error("list selectable voice models", exc) from exc

    async def user_selection(self, owner_person_id: str) -> VoiceUserSelection:
        try:
            async with self._sessions() as session:
                return self._selection(
                    await session.get(VoiceUserPreferenceRecord, owner_person_id)
                )
        except SQLAlchemyError as exc:
            raise _database_error("load voice user preference", exc) from exc

    async def set_user_model(
        self,
        owner_person_id: str,
        selector: str,
    ) -> SelectableVoiceModel:
        provider_alias, separator, model_alias = selector.strip().partition("/")
        if not separator or not provider_alias or not model_alias or "/" in model_alias:
            raise VoiceModelSelectionError("Expected provider/model")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    row = (
                        await session.execute(
                            self._enabled_model_statement().where(
                                VoiceProviderRecord.user_selectable.is_(True),
                                VoiceProviderRecord.alias == provider_alias,
                                VoiceModelRecord.alias == model_alias,
                            )
                        )
                    ).one_or_none()
                    if row is None:
                        raise VoiceModelSelectionError("Voice model is not selectable")
                    model, provider = row
                    await session.execute(
                        postgresql_insert(VoiceUserPreferenceRecord)
                        .values(
                            owner_person_id=owner_person_id,
                            preferred_model_id=model.id,
                        )
                        .on_conflict_do_update(
                            index_elements=[VoiceUserPreferenceRecord.owner_person_id],
                            set_={"preferred_model_id": model.id},
                        )
                    )
                    return SelectableVoiceModel(
                        id=model.id,
                        provider_alias=provider.alias,
                        model_alias=model.alias,
                        display_name=model.display_name,
                        mode=model.mode,
                        selected=True,
                    )
        except SQLAlchemyError as exc:
            raise _database_error("set voice user model", exc) from exc

    async def set_user_voice(self, owner_person_id: str, voice_id: str) -> None:
        normalized = voice_id.strip()
        if not normalized or len(normalized) > 128:
            raise VoiceSpeakerSelectionError
        try:
            async with self._sessions() as session:
                async with session.begin():
                    await session.execute(
                        postgresql_insert(VoiceUserPreferenceRecord)
                        .values(owner_person_id=owner_person_id, voice_id=normalized)
                        .on_conflict_do_update(
                            index_elements=[VoiceUserPreferenceRecord.owner_person_id],
                            set_={"voice_id": normalized},
                        )
                    )
        except SQLAlchemyError as exc:
            raise _database_error("set voice user speaker", exc) from exc

    @staticmethod
    def _enabled_model_statement():
        return (
            select(VoiceModelRecord, VoiceProviderRecord)
            .join(VoiceProviderRecord, VoiceProviderRecord.id == VoiceModelRecord.provider_id)
            .where(
                VoiceModelRecord.enabled.is_(True),
                VoiceProviderRecord.enabled.is_(True),
            )
        )

    @staticmethod
    def _selection(record: VoiceUserPreferenceRecord | None) -> VoiceUserSelection:
        if record is None:
            return VoiceUserSelection()
        return VoiceUserSelection(
            preferred_model_id=record.preferred_model_id,
            voice_id=record.voice_id,
            duplex_mode=record.duplex_mode,
            delegated_agent_model_id=record.delegated_agent_model_id,
        )

    @staticmethod
    def _provider(record: VoiceProviderRecord) -> VoiceProviderConfiguration:
        scheme = urlparse(record.endpoint).scheme.casefold()
        if scheme not in {"ws", "wss"}:
            raise VoiceConfigurationUnavailableError(
                f"Voice Provider {record.alias} requires a WebSocket endpoint"
            )
        return VoiceProviderConfiguration(
            id=record.id,
            alias=record.alias,
            display_name=record.display_name,
            protocol=record.protocol,
            endpoint=record.endpoint,
            credentials=_mapping(record.credentials),
            config=_mapping(record.config),
        )

    @staticmethod
    def _model(record: VoiceModelRecord) -> VoiceModelConfiguration:
        return VoiceModelConfiguration(
            id=record.id,
            provider_id=record.provider_id,
            alias=record.alias,
            remote_model_name=record.remote_model_name,
            display_name=record.display_name,
            mode=record.mode,
            capabilities=_mapping(record.capabilities),
            audio_config=_mapping(record.audio_config),
            prompt_config=_mapping(record.prompt_config),
            limits=_mapping(record.limits),
        )


class SqlAlchemyVoiceSessionRepository:
    """Persist lifecycle transitions and final transcripts in short transactions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        descriptor: VoiceSessionDescriptor,
        configuration: VoiceStartConfiguration,
    ) -> None:
        record = VoiceSessionRecord(
            id=descriptor.session_id,
            owner_person_id=descriptor.owner_person_id,
            area_id=descriptor.voice_channel.area_id,
            voice_channel_id=descriptor.voice_channel.channel_id,
            text_channel_id=descriptor.origin.channel_id,
            model_id=configuration.model.id,
            voice_id=configuration.voice_id,
            duplex_mode=configuration.duplex_mode,
        )
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(record)
        except SQLAlchemyError as exc:
            raise _database_error("create voice session", exc) from exc

    async def mark_active(self, session_id: UUID) -> None:
        await self._update_session(
            update(VoiceSessionRecord)
            .where(VoiceSessionRecord.id == session_id)
            .values(status=PersistedVoiceSessionStatus.ACTIVE),
            "mark voice session active",
        )

    async def finish(
        self,
        session_id: UUID,
        status: PersistedVoiceSessionStatus,
        stop_reason: str,
        *,
        usage: dict[str, Any] | None = None,
        summary: str = "",
    ) -> None:
        if status not in {
            PersistedVoiceSessionStatus.ENDED,
            PersistedVoiceSessionStatus.FAILED,
        }:
            raise ValueError("Voice session finish status must be terminal")
        await self._update_session(
            update(VoiceSessionRecord)
            .where(VoiceSessionRecord.id == session_id)
            .values(
                status=status,
                ended_at=datetime.now(UTC),
                stop_reason=stop_reason[:128],
                usage=dict(usage or {}),
                summary=summary,
            ),
            "finish voice session",
        )

    async def append_final_turn(
        self,
        session_id: UUID,
        sequence: int,
        role: VoiceTurnRole,
        transcript: str,
        *,
        provider_item_id: str = "",
        usage: dict[str, Any] | None = None,
    ) -> None:
        normalized = transcript.strip()
        if sequence <= 0 or not normalized:
            raise ValueError("Final voice turn requires positive sequence and transcript")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(
                        VoiceTurnRecord(
                            session_id=session_id,
                            sequence=sequence,
                            role=role,
                            transcript=normalized,
                            provider_item_id=provider_item_id[:256],
                            usage=dict(usage or {}),
                        )
                    )
        except SQLAlchemyError as exc:
            raise _database_error("append final voice turn", exc) from exc

    async def _update_session(self, statement, operation: str) -> None:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(statement)
                    if result.rowcount != 1:
                        raise DatabaseError(f"Cannot {operation}: voice session was not found")
        except SQLAlchemyError as exc:
            raise _database_error(operation, exc) from exc


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType({str(key): item for key, item in value.items()})


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    return DatabaseError(f"Failed to {operation}: {type(error).__name__}")

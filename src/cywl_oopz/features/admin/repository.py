"""PostgreSQL persistence for idempotent channel initialization."""

from __future__ import annotations

import logging

from sqlalchemy import column, table
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError

from .models import (
    AreaChannelCatalog,
    AreaInitializationResult,
    ChannelInitializationResult,
    ChannelKey,
)

logger = logging.getLogger(__name__)

_TEXT_SETTINGS = table(
    "channel_settings",
    column("area_id"),
    column("channel_id"),
)
_VOICE_SETTINGS = table(
    "voice_channel_settings",
    column("area_id"),
    column("voice_channel_id"),
)


class SqlAlchemyChannelInitializationRepository:
    """Use server defaults and conflict-ignore inserts without overwriting settings."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def initialize_text_channel(
        self,
        channel: ChannelKey,
    ) -> ChannelInitializationResult:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(self._text_insert((channel,)))
                    return ChannelInitializationResult(created=result.rowcount == 1)
        except SQLAlchemyError as exc:
            raise _database_error("initialize text channel", exc) from exc

    async def initialize_area(
        self,
        catalog: AreaChannelCatalog,
    ) -> AreaInitializationResult:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    text_created = await self._insert_text(session, catalog.text_channels)
                    voice_created = await self._insert_voice(session, catalog.voice_channels)
            return AreaInitializationResult(
                text_created=text_created,
                text_existing=len(catalog.text_channels) - text_created,
                voice_created=voice_created,
                voice_existing=len(catalog.voice_channels) - voice_created,
            )
        except SQLAlchemyError as exc:
            raise _database_error("initialize area channels", exc) from exc

    @classmethod
    async def _insert_text(
        cls,
        session: AsyncSession,
        channels: tuple[ChannelKey, ...],
    ) -> int:
        if not channels:
            return 0
        result = await session.execute(cls._text_insert(channels))
        return int(result.rowcount or 0)

    @staticmethod
    async def _insert_voice(
        session: AsyncSession,
        channels: tuple[ChannelKey, ...],
    ) -> int:
        if not channels:
            return 0
        result = await session.execute(
            postgresql_insert(_VOICE_SETTINGS)
            .values(
                [
                    {
                        "area_id": channel.area_id,
                        "voice_channel_id": channel.channel_id,
                    }
                    for channel in channels
                ]
            )
            .on_conflict_do_nothing(
                index_elements=[
                    _VOICE_SETTINGS.c.area_id,
                    _VOICE_SETTINGS.c.voice_channel_id,
                ]
            )
        )
        return int(result.rowcount or 0)

    @staticmethod
    def _text_insert(channels: tuple[ChannelKey, ...]):
        return (
            postgresql_insert(_TEXT_SETTINGS)
            .values(
                [
                    {
                        "area_id": channel.area_id,
                        "channel_id": channel.channel_id,
                    }
                    for channel in channels
                ]
            )
            .on_conflict_do_nothing(
                index_elements=[
                    _TEXT_SETTINGS.c.area_id,
                    _TEXT_SETTINGS.c.channel_id,
                ]
            )
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning("Failed to %s: error=%s", operation, type(error).__name__)
    return DatabaseError(f"Failed to {operation}")

"""Channel-level feature policy repository."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError

from .models import ChannelSettingsRecord


class ChannelSettingsRepository(Protocol):
    """Read boundary for channel feature policy."""

    async def is_chat_enabled(self, area_id: str, channel_id: str) -> bool:
        """Return whether ordinary non-mention messages should trigger text chat."""


class SqlAlchemyChannelSettingsRepository:
    """PostgreSQL implementation of channel-level feature policy."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def is_chat_enabled(self, area_id: str, channel_id: str) -> bool:
        try:
            async with self._sessions() as session:
                enabled = await session.scalar(
                    select(ChannelSettingsRecord.chat_enabled).where(
                        ChannelSettingsRecord.area_id == area_id,
                        ChannelSettingsRecord.channel_id == channel_id,
                    )
                )
                return bool(enabled)
        except SQLAlchemyError as exc:
            raise DatabaseError("Failed to load channel settings") from exc

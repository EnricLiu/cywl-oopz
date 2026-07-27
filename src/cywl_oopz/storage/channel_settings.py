"""Channel-level feature policy repository."""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError

from .models import ChannelSettingsRecord

logger = logging.getLogger(__name__)


class ChannelSettingsRepository(Protocol):
    """Read boundary for channel feature policy."""

    async def is_chat_enabled(self, area_id: str, channel_id: str) -> bool:
        """Return whether ordinary non-mention messages should trigger text chat."""

    async def enabled_agent_tools(
        self,
        area_id: str,
        channel_id: str,
    ) -> frozenset[str]:
        """Return explicitly enabled Agent tool names."""


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
            logger.warning("Failed to load channel chat setting: error=%s", type(exc).__name__)
            raise DatabaseError("Failed to load channel settings") from exc

    async def enabled_agent_tools(
        self,
        area_id: str,
        channel_id: str,
    ) -> frozenset[str]:
        """Load the channel tool allow-list without exposing other settings."""
        try:
            async with self._sessions() as session:
                raw = await session.scalar(
                    select(ChannelSettingsRecord.enabled_agent_tools).where(
                        ChannelSettingsRecord.area_id == area_id,
                        ChannelSettingsRecord.channel_id == channel_id,
                    )
                )
                if not isinstance(raw, list):
                    return frozenset()
                return frozenset(
                    item.strip() for item in raw if isinstance(item, str) and item.strip()
                )
        except SQLAlchemyError as exc:
            logger.warning("Failed to load channel Agent tools: error=%s", type(exc).__name__)
            raise DatabaseError("Failed to load channel Agent tools") from exc

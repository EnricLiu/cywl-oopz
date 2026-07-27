"""Conversation repository port and PostgreSQL implementation."""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import ConversationSessionRecord

from .models import ChatMessage, ConversationKey, ConversationSession

logger = logging.getLogger(__name__)


class ConversationRepository(Protocol):
    """Persistence boundary for expiring chat transcripts."""

    async def get(self, key: ConversationKey) -> ConversationSession | None:
        """Load one session, without deciding whether its TTL has elapsed."""

    async def save(self, session: ConversationSession) -> None:
        """Atomically insert or update one session."""

    async def delete(self, key: ConversationKey) -> None:
        """Delete one session and its persisted message history."""


class SqlAlchemyConversationRepository:
    """PostgreSQL-backed implementation using short-lived ORM sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, key: ConversationKey) -> ConversationSession | None:
        """Load a session by its privacy scope and OOPZ identifiers."""
        try:
            async with self._sessions() as session:
                record = await session.scalar(self._query(key))
                return None if record is None else self._to_domain(record)
        except SQLAlchemyError as exc:
            logger.warning("Failed to load chat conversation session: error=%s", type(exc).__name__)
            raise DatabaseError("Failed to load conversation session") from exc

    async def save(self, chat_session: ConversationSession) -> None:
        """Upsert a session inside one database transaction."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(self._query(chat_session.key))
                    if record is None:
                        session.add(self._to_record(chat_session))
                        return
                    record.selected_model = chat_session.selected_model
                    record.messages = [message.to_payload() for message in chat_session.messages]
                    record.expires_at = chat_session.expires_at
        except SQLAlchemyError as exc:
            logger.warning("Failed to save chat conversation session: error=%s", type(exc).__name__)
            raise DatabaseError("Failed to save conversation session") from exc

    async def delete(self, key: ConversationKey) -> None:
        """Remove one session inside one database transaction."""
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(self._query(key))
                    if record is not None:
                        await session.delete(record)
        except SQLAlchemyError as exc:
            logger.warning(
                "Failed to delete chat conversation session: error=%s", type(exc).__name__
            )
            raise DatabaseError("Failed to delete conversation session") from exc

    @staticmethod
    def _query(key: ConversationKey):
        return select(ConversationSessionRecord).where(
            ConversationSessionRecord.scope == key.scope,
            ConversationSessionRecord.area_id == key.area_id,
            ConversationSessionRecord.channel_id == key.channel_id,
            ConversationSessionRecord.person_id == key.person_id,
        )

    @staticmethod
    def _to_domain(record: ConversationSessionRecord) -> ConversationSession:
        payloads = record.messages if isinstance(record.messages, list) else []
        messages = tuple(
            ChatMessage.from_payload(item) for item in payloads if isinstance(item, dict)
        )
        return ConversationSession(
            key=ConversationKey(
                scope=record.scope,
                area_id=record.area_id,
                channel_id=record.channel_id,
                person_id=record.person_id,
            ),
            messages=messages,
            selected_model=record.selected_model,
            expires_at=record.expires_at,
        )

    @staticmethod
    def _to_record(chat_session: ConversationSession) -> ConversationSessionRecord:
        return ConversationSessionRecord(
            scope=chat_session.key.scope,
            area_id=chat_session.key.area_id,
            channel_id=chat_session.key.channel_id,
            person_id=chat_session.key.person_id,
            selected_model=chat_session.selected_model,
            messages=[message.to_payload() for message in chat_session.messages],
            expires_at=chat_session.expires_at,
        )

"""PostgreSQL persistence for user-controlled long-term Agent memory."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    AgentMemoryItemRecord,
    AgentMemoryPreferenceRecord,
)

from .memory import MemoryItem

logger = logging.getLogger(__name__)


class SqlAlchemyMemoryRepository:
    """Scope every operation by owner and use short-lived sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def preference(self, person_id: str) -> bool | None:
        try:
            async with self._sessions() as session:
                return await session.scalar(
                    select(AgentMemoryPreferenceRecord.enabled).where(
                        AgentMemoryPreferenceRecord.person_id == person_id
                    )
                )
        except SQLAlchemyError as exc:
            raise _database_error("load Agent memory preference", exc) from exc

    async def set_preference(self, person_id: str, enabled: bool) -> None:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.get(AgentMemoryPreferenceRecord, person_id)
                    if record is None:
                        session.add(
                            AgentMemoryPreferenceRecord(
                                person_id=person_id,
                                enabled=enabled,
                            )
                        )
                    else:
                        record.enabled = enabled
        except SQLAlchemyError as exc:
            raise _database_error("save Agent memory preference", exc) from exc

    async def add(self, item: MemoryItem) -> None:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(
                        AgentMemoryItemRecord(
                            id=item.id,
                            owner_person_id=item.owner_person_id,
                            namespace=item.namespace,
                            content=dict(item.content),
                            source_thread_id=item.source_thread_id,
                            source_message_sequence=item.source_message_sequence,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                            last_used_at=item.last_used_at,
                            expires_at=item.expires_at,
                        )
                    )
        except SQLAlchemyError as exc:
            raise _database_error("save Agent memory item", exc) from exc

    async def list_active(
        self,
        person_id: str,
        now: datetime,
        *,
        limit: int,
    ) -> tuple[MemoryItem, ...]:
        try:
            async with self._sessions() as session:
                records = (
                    await session.scalars(
                        select(AgentMemoryItemRecord)
                        .where(
                            AgentMemoryItemRecord.owner_person_id == person_id,
                            or_(
                                AgentMemoryItemRecord.expires_at.is_(None),
                                AgentMemoryItemRecord.expires_at > now,
                            ),
                        )
                        .order_by(AgentMemoryItemRecord.updated_at.desc())
                        .limit(limit)
                    )
                ).all()
                return tuple(self._to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("list Agent memory items", exc) from exc

    async def count_active(self, person_id: str, now: datetime) -> int:
        try:
            async with self._sessions() as session:
                return (
                    await session.scalar(
                        select(func.count(AgentMemoryItemRecord.id)).where(
                            AgentMemoryItemRecord.owner_person_id == person_id,
                            or_(
                                AgentMemoryItemRecord.expires_at.is_(None),
                                AgentMemoryItemRecord.expires_at > now,
                            ),
                        )
                    )
                    or 0
                )
        except SQLAlchemyError as exc:
            raise _database_error("count Agent memory items", exc) from exc

    async def delete(self, person_id: str, item_id: UUID) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        delete(AgentMemoryItemRecord).where(
                            AgentMemoryItemRecord.id == item_id,
                            AgentMemoryItemRecord.owner_person_id == person_id,
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("delete Agent memory item", exc) from exc

    async def delete_all(self, person_id: str) -> int:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        delete(AgentMemoryItemRecord).where(
                            AgentMemoryItemRecord.owner_person_id == person_id
                        )
                    )
                    return result.rowcount
        except SQLAlchemyError as exc:
            raise _database_error("delete Agent memory items", exc) from exc

    async def touch(
        self,
        person_id: str,
        item_ids: tuple[UUID, ...],
        now: datetime,
    ) -> None:
        if not item_ids:
            return
        try:
            async with self._sessions() as session:
                async with session.begin():
                    await session.execute(
                        update(AgentMemoryItemRecord)
                        .where(
                            AgentMemoryItemRecord.owner_person_id == person_id,
                            AgentMemoryItemRecord.id.in_(item_ids),
                        )
                        .values(last_used_at=now)
                    )
        except SQLAlchemyError as exc:
            raise _database_error("touch Agent memory items", exc) from exc

    @staticmethod
    def _to_domain(record: AgentMemoryItemRecord) -> MemoryItem:
        content = record.content if isinstance(record.content, dict) else {}
        return MemoryItem(
            id=record.id,
            owner_person_id=record.owner_person_id,
            namespace=record.namespace,
            content=content,
            source_thread_id=record.source_thread_id,
            source_message_sequence=record.source_message_sequence,
            created_at=record.created_at,
            updated_at=record.updated_at,
            last_used_at=record.last_used_at,
            expires_at=record.expires_at,
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    """Report a static operation name without rendering SQL or record data."""
    logger.warning(
        "Agent memory persistence failed: operation=%s error=%s",
        operation,
        type(error).__name__,
    )
    return DatabaseError(f"Failed to {operation}")

"""PostgreSQL persistence for Agent skill bundles."""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    AgentSkillCatalogStateRecord,
    AgentSkillRecord,
    AgentSkillResourceRecord,
)

from .models import AgentSkill, AgentSkillResource

logger = logging.getLogger(__name__)


class SqlAlchemyAgentSkillRepository:
    """Load each immutable catalog candidate inside one short-lived session."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        try:
            async with self._sessions() as session:
                skill_records = (
                    await session.scalars(
                        select(AgentSkillRecord)
                        .where(AgentSkillRecord.enabled.is_(True))
                        .order_by(AgentSkillRecord.name)
                    )
                ).all()
                skill_ids = tuple(record.id for record in skill_records)
                resource_records = (
                    (
                        await session.scalars(
                            select(AgentSkillResourceRecord)
                            .where(AgentSkillResourceRecord.skill_id.in_(skill_ids))
                            .order_by(
                                AgentSkillResourceRecord.skill_id,
                                AgentSkillResourceRecord.position,
                            )
                        )
                    ).all()
                    if skill_ids
                    else []
                )
        except SQLAlchemyError as exc:
            raise _database_error("load Agent skills", exc) from exc

        resources_by_skill: dict[UUID, list[AgentSkillResource]] = defaultdict(list)
        for record in resource_records:
            resources_by_skill[record.skill_id].append(
                AgentSkillResource(
                    id=record.id,
                    key=record.key,
                    display_name=record.display_name,
                    description=record.description,
                    kind=record.kind,
                    media_type=record.media_type,
                    content=record.content,
                    position=record.position,
                )
            )
        return tuple(
            AgentSkill(
                id=record.id,
                name=record.name,
                display_name=record.display_name,
                description=record.description,
                instructions=record.instructions,
                version=record.version,
                revision=record.revision,
                required_tools=frozenset(record.required_tools),
                resources=tuple(resources_by_skill[record.id]),
                metadata=record.skill_metadata,
            )
            for record in skill_records
        )

    async def generation(self) -> int:
        try:
            async with self._sessions() as session:
                generation = await session.scalar(
                    select(AgentSkillCatalogStateRecord.generation).where(
                        AgentSkillCatalogStateRecord.singleton_id == 1
                    )
                )
        except SQLAlchemyError as exc:
            raise _database_error("load Agent skill catalog generation", exc) from exc
        if generation is None:
            raise DatabaseError("Agent skill catalog state is missing")
        return generation


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning(
        "Agent skill persistence failed: operation=%s error=%s",
        operation,
        type(error).__name__,
    )
    return DatabaseError(f"Failed to {operation}")

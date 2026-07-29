"""PostgreSQL persistence for Agent skill bundles."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, delete, exists, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    AgentSkillCatalogStateRecord,
    AgentSkillRecord,
    AgentSkillResourceRecord,
    AgentSkillShareRecord,
)

from .errors import AgentSkillConflictError, AgentSkillNotFoundError
from .models import (
    AgentSkill,
    AgentSkillResource,
    AgentSkillShare,
    SkillOwnershipKind,
    SkillShareStatus,
)

logger = logging.getLogger(__name__)


class SqlAlchemyAgentSkillRepository:
    """Persist builtin and user-owned Skill data in short transactions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        """Load only builtin Skills for the legacy global catalog."""
        return await self._load_bundles(
            AgentSkillRecord.enabled.is_(True),
            AgentSkillRecord.ownership_kind == SkillOwnershipKind.BUILTIN,
        )

    async def load_accessible(self, person_id: str) -> tuple[AgentSkill, ...]:
        """Load active builtin, owned, and accepted shared bundles for one user."""
        recipient = person_id.strip()
        if not recipient:
            raise ValueError("Skill library person ID must not be empty")
        accepted_share = exists(
            select(AgentSkillShareRecord.id).where(
                AgentSkillShareRecord.skill_id == AgentSkillRecord.id,
                AgentSkillShareRecord.recipient_person_id == recipient,
                AgentSkillShareRecord.status == SkillShareStatus.ACCEPTED,
            )
        )
        return await self._load_bundles(
            AgentSkillRecord.enabled.is_(True),
            AgentSkillRecord.archived_at.is_(None),
            or_(
                AgentSkillRecord.ownership_kind == SkillOwnershipKind.BUILTIN,
                AgentSkillRecord.owner_person_id == recipient,
                and_(
                    AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                    accepted_share,
                ),
            ),
        )

    async def add_personal(self, skill: AgentSkill) -> None:
        """Persist one complete personal Skill and its resources atomically."""
        if skill.ownership_kind is not SkillOwnershipKind.PERSONAL:
            raise ValueError("Only personal Skills may be added to a user library")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(self._skill_record(skill))
                    session.add_all(
                        AgentSkillResourceRecord(
                            id=resource.id,
                            skill_id=skill.id,
                            key=resource.key,
                            display_name=resource.display_name,
                            description=resource.description,
                            kind=resource.kind,
                            media_type=resource.media_type,
                            content=resource.content,
                            position=resource.position,
                        )
                        for resource in skill.resources
                    )
        except IntegrityError as exc:
            raise AgentSkillConflictError("Personal Skill conflicts with existing data") from exc
        except SQLAlchemyError as exc:
            raise _database_error("create personal Agent skill", exc) from exc

    async def get_owned(self, person_id: str, skill_id: UUID) -> AgentSkill | None:
        """Load one complete Skill only when the caller owns it."""
        owner = _person_id(person_id, "Skill owner")
        skills = await self._load_bundles(
            AgentSkillRecord.id == skill_id,
            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
            AgentSkillRecord.owner_person_id == owner,
        )
        return skills[0] if skills else None

    async def invite(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_id: str,
        now: datetime,
    ) -> AgentSkillShare:
        """Create, retain, or refresh one invitation under an owner lock."""
        owner = _person_id(owner_person_id, "Skill owner")
        recipient = _person_id(recipient_person_id, "Skill share recipient")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    owned = await session.scalar(
                        select(AgentSkillRecord)
                        .where(
                            AgentSkillRecord.id == skill_id,
                            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                            AgentSkillRecord.owner_person_id == owner,
                        )
                        .with_for_update()
                    )
                    if owned is None:
                        raise AgentSkillNotFoundError("Owned Skill was not found")
                    record = await session.scalar(
                        select(AgentSkillShareRecord)
                        .where(
                            AgentSkillShareRecord.skill_id == skill_id,
                            AgentSkillShareRecord.recipient_person_id == recipient,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        record = AgentSkillShareRecord(
                            skill_id=skill_id,
                            recipient_person_id=recipient,
                            status=SkillShareStatus.PENDING,
                            created_at=now,
                            updated_at=now,
                        )
                        session.add(record)
                    elif record.status is SkillShareStatus.DECLINED:
                        record.status = SkillShareStatus.PENDING
                        record.responded_at = None
                    await session.flush()
                    await session.refresh(record)
                    result = self._to_share(record)
        except AgentSkillNotFoundError:
            raise
        except IntegrityError as exc:
            raise AgentSkillConflictError("Skill invitation conflicts with current state") from exc
        except SQLAlchemyError as exc:
            raise _database_error("invite Agent skill recipient", exc) from exc
        return result

    async def respond(
        self,
        recipient_person_id: str,
        share_id: UUID,
        status: SkillShareStatus,
        now: datetime,
    ) -> AgentSkillShare:
        """Accept or decline one invitation owned by the recipient."""
        if status is SkillShareStatus.PENDING:
            raise ValueError("Skill invitation response must be accepted or declined")
        recipient = _person_id(recipient_person_id, "Skill share recipient")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(
                        select(AgentSkillShareRecord)
                        .where(
                            AgentSkillShareRecord.id == share_id,
                            AgentSkillShareRecord.recipient_person_id == recipient,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        raise AgentSkillNotFoundError("Skill invitation was not found")
                    if record.status is not status:
                        record.status = status
                        record.responded_at = now
                    await session.flush()
                    await session.refresh(record)
                    result = self._to_share(record)
        except AgentSkillNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("respond to Agent skill invitation", exc) from exc
        return result

    async def revoke(
        self,
        owner_person_id: str,
        share_id: UUID,
    ) -> bool:
        """Delete one share only through its owning personal Skill."""
        owner = _person_id(owner_person_id, "Skill owner")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await session.scalar(
                        select(AgentSkillShareRecord)
                        .join(
                            AgentSkillRecord,
                            AgentSkillRecord.id == AgentSkillShareRecord.skill_id,
                        )
                        .where(
                            AgentSkillShareRecord.id == share_id,
                            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                            AgentSkillRecord.owner_person_id == owner,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        return False
                    await session.execute(
                        delete(AgentSkillShareRecord).where(AgentSkillShareRecord.id == record.id)
                    )
        except SQLAlchemyError as exc:
            raise _database_error("revoke Agent skill share", exc) from exc
        return True

    async def _load_bundles(
        self,
        *predicates: ColumnElement[bool],
    ) -> tuple[AgentSkill, ...]:
        try:
            async with self._sessions() as session:
                skill_records = (
                    await session.scalars(
                        select(AgentSkillRecord)
                        .where(*predicates)
                        .order_by(
                            AgentSkillRecord.ownership_kind,
                            AgentSkillRecord.name,
                            AgentSkillRecord.id,
                        )
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

        return self._to_skills(skill_records, resource_records)

    @staticmethod
    def _to_skills(
        skill_records: list[AgentSkillRecord],
        resource_records: list[AgentSkillResourceRecord],
    ) -> tuple[AgentSkill, ...]:
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
                ownership_kind=record.ownership_kind,
                owner_person_id=record.owner_person_id,
                enabled=record.enabled,
                archived_at=record.archived_at,
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

    @staticmethod
    def _skill_record(skill: AgentSkill) -> AgentSkillRecord:
        return AgentSkillRecord(
            id=skill.id,
            name=skill.name,
            display_name=skill.display_name,
            description=skill.description,
            instructions=skill.instructions,
            version=skill.version,
            revision=skill.revision,
            required_tools=sorted(skill.required_tools),
            skill_metadata=_thaw_json_object(skill.metadata),
            enabled=skill.enabled,
            ownership_kind=skill.ownership_kind,
            owner_person_id=skill.owner_person_id,
            archived_at=skill.archived_at,
        )

    @staticmethod
    def _to_share(record: AgentSkillShareRecord) -> AgentSkillShare:
        return AgentSkillShare(
            id=record.id,
            skill_id=record.skill_id,
            recipient_person_id=record.recipient_person_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            responded_at=record.responded_at,
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning(
        "Agent skill persistence failed: operation=%s error=%s",
        operation,
        type(error).__name__,
    )
    return DatabaseError(f"Failed to {operation}")


def _person_id(value: str, label: str) -> str:
    person_id = value.strip()
    if not person_id:
        raise ValueError(f"{label} ID must not be empty")
    return person_id


def _thaw_json_object(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _thaw_json(item) for key, item in value.items()}


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _thaw_json_object(value)
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value

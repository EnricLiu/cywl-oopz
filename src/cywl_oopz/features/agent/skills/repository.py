"""PostgreSQL persistence for Agent skill bundles."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, case, delete, exists, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    AgentSkillRecord,
    AgentSkillResourceRecord,
    AgentSkillShareRecord,
)

from .errors import (
    AgentSkillConflictError,
    AgentSkillNotFoundError,
    AgentSkillRevisionConflictError,
)
from .models import (
    AgentSkill,
    AgentSkillBundle,
    AgentSkillDiscovery,
    AgentSkillInspection,
    AgentSkillOutgoingShare,
    AgentSkillOwnedSummary,
    AgentSkillResource,
    AgentSkillResourceManifest,
    AgentSkillShare,
    AgentSkillShareSummary,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillShareStatus,
)

logger = logging.getLogger(__name__)


class SqlAlchemyAgentSkillRepository:
    """Persist builtin and user-owned Skill data in short transactions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_accessible(
        self,
        person_id: str,
    ) -> tuple[AgentSkillDiscovery, ...]:
        """Query only metadata needed for one caller's run discovery."""
        recipient = _person_id(person_id, "Skill library person")
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(*self._discovery_columns())
                        .where(*self._accessible_predicates(recipient))
                        .order_by(
                            case(
                                (
                                    AgentSkillRecord.owner_person_id == recipient,
                                    0,
                                ),
                                (
                                    AgentSkillRecord.ownership_kind == SkillOwnershipKind.BUILTIN,
                                    1,
                                ),
                                else_=2,
                            ),
                            AgentSkillRecord.name,
                            AgentSkillRecord.id,
                        )
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("list accessible Agent skills", exc) from exc
        return tuple(self._to_discovery(row, recipient) for row in rows)

    async def load_accessible_bundle(
        self,
        person_id: str,
        skill_id: UUID,
        revision: int,
    ) -> AgentSkillBundle | None:
        """Load instructions and resource manifests without resource bodies."""
        recipient = _person_id(person_id, "Skill library person")
        try:
            async with self._sessions() as session:
                row = (
                    await session.execute(
                        select(
                            *self._discovery_columns(),
                            AgentSkillRecord.instructions,
                        ).where(
                            AgentSkillRecord.id == skill_id,
                            *self._accessible_predicates(recipient),
                        )
                    )
                ).one_or_none()
                if row is None:
                    return None
                discovery = self._to_discovery(row, recipient)
                if discovery.revision != revision:
                    raise AgentSkillRevisionConflictError(
                        "Agent Skill changed after this run started"
                    )
                resources = (
                    await session.execute(
                        select(
                            AgentSkillResourceRecord.id,
                            AgentSkillResourceRecord.key,
                            AgentSkillResourceRecord.display_name,
                            AgentSkillResourceRecord.description,
                            AgentSkillResourceRecord.kind,
                            AgentSkillResourceRecord.media_type,
                            AgentSkillResourceRecord.position,
                        )
                        .where(AgentSkillResourceRecord.skill_id == skill_id)
                        .order_by(AgentSkillResourceRecord.position)
                    )
                ).all()
        except AgentSkillRevisionConflictError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("load accessible Agent skill bundle", exc) from exc
        return AgentSkillBundle(
            discovery=discovery,
            instructions=row[9],
            resources=tuple(
                AgentSkillResourceManifest(
                    id=resource.id,
                    key=resource.key,
                    display_name=resource.display_name,
                    description=resource.description,
                    kind=resource.kind,
                    media_type=resource.media_type,
                    position=resource.position,
                )
                for resource in resources
            ),
        )

    async def read_accessible_resource(
        self,
        person_id: str,
        skill_id: UUID,
        resource_id: UUID,
        revision: int,
    ) -> AgentSkillResource | None:
        """Load one resource body only after rechecking caller access and revision."""
        recipient = _person_id(person_id, "Skill library person")
        try:
            async with self._sessions() as session:
                current_revision = await session.scalar(
                    select(AgentSkillRecord.revision).where(
                        AgentSkillRecord.id == skill_id,
                        *self._accessible_predicates(recipient),
                    )
                )
                if current_revision is None:
                    return None
                if current_revision != revision:
                    raise AgentSkillRevisionConflictError(
                        "Agent Skill changed after this run started"
                    )
                resource = (
                    await session.execute(
                        select(
                            AgentSkillResourceRecord.id,
                            AgentSkillResourceRecord.key,
                            AgentSkillResourceRecord.display_name,
                            AgentSkillResourceRecord.description,
                            AgentSkillResourceRecord.kind,
                            AgentSkillResourceRecord.media_type,
                            AgentSkillResourceRecord.content,
                            AgentSkillResourceRecord.position,
                        ).where(
                            AgentSkillResourceRecord.skill_id == skill_id,
                            AgentSkillResourceRecord.id == resource_id,
                        )
                    )
                ).one_or_none()
        except AgentSkillRevisionConflictError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("read accessible Agent skill resource", exc) from exc
        if resource is None:
            return None
        return AgentSkillResource(
            id=resource.id,
            key=resource.key,
            display_name=resource.display_name,
            description=resource.description,
            kind=resource.kind,
            media_type=resource.media_type,
            content=resource.content,
            position=resource.position,
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

    async def list_owned(
        self,
        person_id: str,
    ) -> tuple[AgentSkillOwnedSummary, ...]:
        """List metadata for active and archived Skills owned by one user."""
        owner = _person_id(person_id, "Skill owner")
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(
                            *self._discovery_columns(),
                            AgentSkillRecord.enabled,
                            AgentSkillRecord.archived_at,
                        )
                        .where(
                            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                            AgentSkillRecord.owner_person_id == owner,
                        )
                        .order_by(AgentSkillRecord.name, AgentSkillRecord.id)
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("list owned Agent skills", exc) from exc
        return tuple(
            AgentSkillOwnedSummary(
                self._to_discovery(row, owner),
                active=bool(row[9]) and row[10] is None,
            )
            for row in rows
        )

    async def inspect_accessible(
        self,
        person_id: str,
        skill_id: UUID,
    ) -> AgentSkillInspection | None:
        """Load fresh instructions/manifests, including archived owned content."""
        caller = _person_id(person_id, "Skill library person")
        owned = and_(
            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
            AgentSkillRecord.owner_person_id == caller,
        )
        active_access = and_(*self._accessible_predicates(caller))
        try:
            async with self._sessions() as session:
                row = (
                    await session.execute(
                        select(
                            *self._discovery_columns(),
                            AgentSkillRecord.instructions,
                            AgentSkillRecord.enabled,
                            AgentSkillRecord.archived_at,
                        ).where(
                            AgentSkillRecord.id == skill_id,
                            or_(owned, active_access),
                        )
                    )
                ).one_or_none()
                if row is None:
                    return None
                resources = (
                    await session.execute(
                        select(
                            AgentSkillResourceRecord.id,
                            AgentSkillResourceRecord.key,
                            AgentSkillResourceRecord.display_name,
                            AgentSkillResourceRecord.description,
                            AgentSkillResourceRecord.kind,
                            AgentSkillResourceRecord.media_type,
                            AgentSkillResourceRecord.position,
                        )
                        .where(AgentSkillResourceRecord.skill_id == skill_id)
                        .order_by(AgentSkillResourceRecord.position)
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("inspect accessible Agent skill", exc) from exc
        return AgentSkillInspection(
            AgentSkillBundle(
                discovery=self._to_discovery(row, caller),
                instructions=row[9],
                resources=tuple(
                    AgentSkillResourceManifest(
                        id=resource.id,
                        key=resource.key,
                        display_name=resource.display_name,
                        description=resource.description,
                        kind=resource.kind,
                        media_type=resource.media_type,
                        position=resource.position,
                    )
                    for resource in resources
                ),
            ),
            active=bool(row[10]) and row[11] is None,
        )

    async def read_inspectable_resource(
        self,
        person_id: str,
        skill_id: UUID,
        resource_key: str,
    ) -> AgentSkillResource | None:
        """Read one resource body visible to fresh management inspection."""
        caller = _person_id(person_id, "Skill library person")
        owned = and_(
            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
            AgentSkillRecord.owner_person_id == caller,
        )
        active_access = and_(*self._accessible_predicates(caller))
        try:
            async with self._sessions() as session:
                row = (
                    await session.execute(
                        select(
                            AgentSkillResourceRecord.id,
                            AgentSkillResourceRecord.key,
                            AgentSkillResourceRecord.display_name,
                            AgentSkillResourceRecord.description,
                            AgentSkillResourceRecord.kind,
                            AgentSkillResourceRecord.media_type,
                            AgentSkillResourceRecord.content,
                            AgentSkillResourceRecord.position,
                        )
                        .join(
                            AgentSkillRecord,
                            AgentSkillRecord.id == AgentSkillResourceRecord.skill_id,
                        )
                        .where(
                            AgentSkillRecord.id == skill_id,
                            AgentSkillResourceRecord.key == resource_key,
                            or_(owned, active_access),
                        )
                    )
                ).one_or_none()
        except SQLAlchemyError as exc:
            raise _database_error("inspect Agent skill resource", exc) from exc
        if row is None:
            return None
        return AgentSkillResource(
            id=row.id,
            key=row.key,
            display_name=row.display_name,
            description=row.description,
            kind=row.kind,
            media_type=row.media_type,
            content=row.content,
            position=row.position,
        )

    async def update_owned(
        self,
        skill: AgentSkill,
        expected_revision: int,
    ) -> AgentSkill:
        """Replace core fields under an owner/revision row lock."""
        if skill.ownership_kind is not SkillOwnershipKind.PERSONAL:
            raise ValueError("Only personal Skills may be updated")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await self._locked_owned(
                        session,
                        skill.owner_person_id or "",
                        skill.id,
                        expected_revision,
                    )
                    record.name = skill.name
                    record.display_name = skill.display_name
                    record.description = skill.description
                    record.instructions = skill.instructions
                    record.version = skill.version
                    record.required_tools = sorted(skill.required_tools)
                    record.skill_metadata = _thaw_json_object(skill.metadata)
                    await session.flush()
                    await session.refresh(record)
                    result = await self._owned_bundle(session, record)
        except (AgentSkillNotFoundError, AgentSkillRevisionConflictError):
            raise
        except IntegrityError as exc:
            raise AgentSkillConflictError("Personal Skill conflicts with existing data") from exc
        except SQLAlchemyError as exc:
            raise _database_error("update personal Agent skill", exc) from exc
        return result

    async def upsert_owned_resource(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        resource: AgentSkillResource,
    ) -> AgentSkill:
        """Insert or replace one resource and return the trigger-bumped parent."""
        owner = _person_id(owner_person_id, "Skill owner")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    skill_record = await self._locked_owned(
                        session,
                        owner,
                        skill_id,
                        expected_revision,
                    )
                    record = await session.scalar(
                        select(AgentSkillResourceRecord)
                        .where(
                            AgentSkillResourceRecord.skill_id == skill_id,
                            AgentSkillResourceRecord.key == resource.key,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        record = AgentSkillResourceRecord(
                            id=resource.id,
                            skill_id=skill_id,
                            key=resource.key,
                            display_name=resource.display_name,
                            description=resource.description,
                            kind=resource.kind,
                            media_type=resource.media_type,
                            content=resource.content,
                            position=resource.position,
                        )
                        session.add(record)
                    else:
                        record.display_name = resource.display_name
                        record.description = resource.description
                        record.kind = resource.kind
                        record.media_type = resource.media_type
                        record.content = resource.content
                        record.position = resource.position
                    await session.flush()
                    await session.refresh(skill_record)
                    result = await self._owned_bundle(session, skill_record)
        except (AgentSkillNotFoundError, AgentSkillRevisionConflictError):
            raise
        except IntegrityError as exc:
            raise AgentSkillConflictError("Skill resource conflicts with existing data") from exc
        except SQLAlchemyError as exc:
            raise _database_error("upsert personal Agent skill resource", exc) from exc
        return result

    async def remove_owned_resource(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        resource_key: str,
    ) -> AgentSkill:
        """Remove one owned resource and return the trigger-bumped parent."""
        owner = _person_id(owner_person_id, "Skill owner")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    skill_record = await self._locked_owned(
                        session,
                        owner,
                        skill_id,
                        expected_revision,
                    )
                    record = await session.scalar(
                        select(AgentSkillResourceRecord)
                        .where(
                            AgentSkillResourceRecord.skill_id == skill_id,
                            AgentSkillResourceRecord.key == resource_key,
                        )
                        .with_for_update()
                    )
                    if record is None:
                        raise AgentSkillNotFoundError("Owned Skill resource was not found")
                    await session.delete(record)
                    await session.flush()
                    await session.refresh(skill_record)
                    result = await self._owned_bundle(session, skill_record)
        except (AgentSkillNotFoundError, AgentSkillRevisionConflictError):
            raise
        except SQLAlchemyError as exc:
            raise _database_error("remove personal Agent skill resource", exc) from exc
        return result

    async def set_owned_state(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        *,
        enabled: bool,
        archived_at: datetime | None,
    ) -> AgentSkill:
        """Archive or restore one Skill and return its new revision."""
        owner = _person_id(owner_person_id, "Skill owner")
        if enabled == (archived_at is not None):
            raise ValueError("Archived Skill state is inconsistent")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    record = await self._locked_owned(
                        session,
                        owner,
                        skill_id,
                        expected_revision,
                    )
                    record.enabled = enabled
                    record.archived_at = archived_at
                    await session.flush()
                    await session.refresh(record)
                    result = await self._owned_bundle(session, record)
        except (AgentSkillNotFoundError, AgentSkillRevisionConflictError):
            raise
        except SQLAlchemyError as exc:
            raise _database_error("set personal Agent skill state", exc) from exc
        return result

    async def pending_invitations(
        self,
        recipient_person_id: str,
    ) -> tuple[AgentSkillShareSummary, ...]:
        """List pending invitations with metadata but without Skill content."""
        recipient = _person_id(recipient_person_id, "Skill share recipient")
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(
                            AgentSkillShareRecord,
                            *self._discovery_columns(),
                            AgentSkillRecord.enabled,
                            AgentSkillRecord.archived_at,
                        )
                        .join(
                            AgentSkillRecord,
                            AgentSkillRecord.id == AgentSkillShareRecord.skill_id,
                        )
                        .where(
                            AgentSkillShareRecord.recipient_person_id == recipient,
                            AgentSkillShareRecord.status == SkillShareStatus.PENDING,
                        )
                        .order_by(
                            AgentSkillShareRecord.created_at,
                            AgentSkillShareRecord.id,
                        )
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("list Agent skill invitations", exc) from exc
        return tuple(self._to_share_summary(row, recipient) for row in rows)

    async def invite_many(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_ids: tuple[str, ...],
        now: datetime,
    ) -> tuple[AgentSkillShare, ...]:
        """Create or refresh several invitations in one transaction."""
        owner = _person_id(owner_person_id, "Skill owner")
        recipients = tuple(
            dict.fromkeys(
                _person_id(value, "Skill share recipient") for value in recipient_person_ids
            )
        )
        if not recipients:
            raise ValueError("Skill share recipients must not be empty")
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
                    existing = (
                        await session.scalars(
                            select(AgentSkillShareRecord)
                            .where(
                                AgentSkillShareRecord.skill_id == skill_id,
                                AgentSkillShareRecord.recipient_person_id.in_(recipients),
                            )
                            .with_for_update()
                        )
                    ).all()
                    by_recipient = {record.recipient_person_id: record for record in existing}
                    records: list[AgentSkillShareRecord] = []
                    for recipient in recipients:
                        record = by_recipient.get(recipient)
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
                        records.append(record)
                    await session.flush()
                    for record in records:
                        await session.refresh(record)
                    result = tuple(self._to_share(record) for record in records)
        except AgentSkillNotFoundError:
            raise
        except IntegrityError as exc:
            raise AgentSkillConflictError("Skill invitations conflict with current state") from exc
        except SQLAlchemyError as exc:
            raise _database_error("invite Agent skill recipients", exc) from exc
        return result

    async def outgoing_shares(
        self,
        owner_person_id: str,
    ) -> tuple[AgentSkillOutgoingShare, ...]:
        """Aggregate owner shares without returning recipient identities."""
        owner = _person_id(owner_person_id, "Skill owner")
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(
                            AgentSkillShareRecord.status,
                            *self._discovery_columns(),
                            AgentSkillRecord.enabled,
                            AgentSkillRecord.archived_at,
                        )
                        .join(
                            AgentSkillRecord,
                            AgentSkillRecord.id == AgentSkillShareRecord.skill_id,
                        )
                        .where(
                            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                            AgentSkillRecord.owner_person_id == owner,
                        )
                        .order_by(AgentSkillRecord.name, AgentSkillRecord.id)
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("list outgoing Agent skill shares", exc) from exc
        grouped: dict[UUID, tuple[AgentSkillDiscovery, int, int, bool]] = {}
        for row in rows:
            discovery = self._to_discovery(row[1:], owner)
            active = bool(row[10]) and row[11] is None
            current = grouped.get(discovery.id, (discovery, 0, 0, active))
            pending = current[1] + int(row[0] is SkillShareStatus.PENDING)
            accepted = current[2] + int(row[0] is SkillShareStatus.ACCEPTED)
            grouped[discovery.id] = (discovery, pending, accepted, active)
        return tuple(
            AgentSkillOutgoingShare(discovery, pending, accepted, active)
            for discovery, pending, accepted, active in grouped.values()
        )

    async def share_for_recipient(
        self,
        recipient_person_id: str,
        share_id: UUID,
    ) -> AgentSkillShareSummary | None:
        """Read one share only through its recipient scope."""
        recipient = _person_id(recipient_person_id, "Skill share recipient")
        return await self._share_summary(
            AgentSkillShareRecord.id == share_id,
            AgentSkillShareRecord.recipient_person_id == recipient,
            caller_person_id=recipient,
        )

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
                    if (
                        record.status is not SkillShareStatus.PENDING
                        and record.status is not status
                    ):
                        raise AgentSkillConflictError("Skill invitation has already been answered")
                    if record.status is not status:
                        record.status = status
                        record.responded_at = now
                    await session.flush()
                    await session.refresh(record)
                    result = self._to_share(record)
        except (AgentSkillConflictError, AgentSkillNotFoundError):
            raise
        except SQLAlchemyError as exc:
            raise _database_error("respond to Agent skill invitation", exc) from exc
        return result

    async def revoke_owned_shares(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_ids: tuple[str, ...] | None,
    ) -> tuple[AgentSkillShare, ...]:
        """Delete selected or all shares through one owner-scoped Skill lock."""
        owner = _person_id(owner_person_id, "Skill owner")
        recipients = (
            tuple(
                dict.fromkeys(
                    _person_id(value, "Skill share recipient") for value in recipient_person_ids
                )
            )
            if recipient_person_ids is not None
            else None
        )
        if recipients == ():
            raise ValueError("Skill share recipients must not be empty")
        try:
            async with self._sessions() as session:
                async with session.begin():
                    owned = await session.scalar(
                        select(AgentSkillRecord.id)
                        .where(
                            AgentSkillRecord.id == skill_id,
                            AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                            AgentSkillRecord.owner_person_id == owner,
                        )
                        .with_for_update()
                    )
                    if owned is None:
                        raise AgentSkillNotFoundError("Owned Skill was not found")
                    statement = (
                        select(AgentSkillShareRecord)
                        .where(AgentSkillShareRecord.skill_id == skill_id)
                        .order_by(AgentSkillShareRecord.created_at, AgentSkillShareRecord.id)
                        .with_for_update()
                    )
                    if recipients is not None:
                        statement = statement.where(
                            AgentSkillShareRecord.recipient_person_id.in_(recipients)
                        )
                    records = (await session.scalars(statement)).all()
                    result = tuple(self._to_share(record) for record in records)
                    if records:
                        await session.execute(
                            delete(AgentSkillShareRecord).where(
                                AgentSkillShareRecord.id.in_(tuple(record.id for record in records))
                            )
                        )
        except AgentSkillNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("revoke Agent skill shares", exc) from exc
        return result

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

    @staticmethod
    def _discovery_columns() -> tuple[object, ...]:
        return (
            AgentSkillRecord.id,
            AgentSkillRecord.name,
            AgentSkillRecord.display_name,
            AgentSkillRecord.description,
            AgentSkillRecord.version,
            AgentSkillRecord.revision,
            AgentSkillRecord.required_tools,
            AgentSkillRecord.ownership_kind,
            AgentSkillRecord.owner_person_id,
        )

    @staticmethod
    def _accessible_predicates(recipient: str) -> tuple[ColumnElement[bool], ...]:
        accepted_share = exists(
            select(AgentSkillShareRecord.id).where(
                AgentSkillShareRecord.skill_id == AgentSkillRecord.id,
                AgentSkillShareRecord.recipient_person_id == recipient,
                AgentSkillShareRecord.status == SkillShareStatus.ACCEPTED,
            )
        )
        return (
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

    @staticmethod
    def _to_discovery(
        row: Sequence[object],
        recipient: str,
    ) -> AgentSkillDiscovery:
        values = tuple(row)
        ownership = values[7]
        owner_person_id = values[8]
        access = (
            SkillAccessKind.BUILTIN
            if ownership is SkillOwnershipKind.BUILTIN
            else (SkillAccessKind.OWNED if owner_person_id == recipient else SkillAccessKind.SHARED)
        )
        return AgentSkillDiscovery(
            id=values[0],
            name=values[1],
            display_name=values[2],
            description=values[3],
            version=values[4],
            revision=values[5],
            required_tools=frozenset(values[6]),
            access=access,
        )

    @staticmethod
    async def _locked_owned(
        session: AsyncSession,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
    ) -> AgentSkillRecord:
        record = await session.scalar(
            select(AgentSkillRecord)
            .where(
                AgentSkillRecord.id == skill_id,
                AgentSkillRecord.ownership_kind == SkillOwnershipKind.PERSONAL,
                AgentSkillRecord.owner_person_id == owner_person_id,
            )
            .with_for_update()
        )
        if record is None:
            raise AgentSkillNotFoundError("Owned Skill was not found")
        if record.revision != expected_revision:
            raise AgentSkillRevisionConflictError("Owned Skill revision changed")
        return record

    async def _owned_bundle(
        self,
        session: AsyncSession,
        record: AgentSkillRecord,
    ) -> AgentSkill:
        resources = (
            await session.scalars(
                select(AgentSkillResourceRecord)
                .where(AgentSkillResourceRecord.skill_id == record.id)
                .order_by(AgentSkillResourceRecord.position)
            )
        ).all()
        return self._to_skills([record], list(resources))[0]

    async def _share_summary(
        self,
        *predicates: ColumnElement[bool],
        caller_person_id: str,
    ) -> AgentSkillShareSummary | None:
        try:
            async with self._sessions() as session:
                row = (
                    await session.execute(
                        select(
                            AgentSkillShareRecord,
                            *self._discovery_columns(),
                            AgentSkillRecord.enabled,
                            AgentSkillRecord.archived_at,
                        )
                        .join(
                            AgentSkillRecord,
                            AgentSkillRecord.id == AgentSkillShareRecord.skill_id,
                        )
                        .where(*predicates)
                    )
                ).one_or_none()
        except SQLAlchemyError as exc:
            raise _database_error("load Agent skill share", exc) from exc
        if row is None:
            return None
        return self._to_share_summary(row, caller_person_id)

    @classmethod
    def _to_share_summary(
        cls,
        row: Sequence[object],
        caller_person_id: str,
    ) -> AgentSkillShareSummary:
        return AgentSkillShareSummary(
            cls._to_share(row[0]),
            cls._to_discovery(row[1:], caller_person_id),
            active=bool(row[10]) and row[11] is None,
        )

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

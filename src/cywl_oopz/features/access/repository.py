"""PostgreSQL role-binding repository with short async transactions."""

from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import RbacRoleBindingRecord

from .models import AccessRole, RoleBinding, RoleBindingScope

logger = logging.getLogger(__name__)


class SqlAlchemyRoleBindingRepository:
    """Load role changes immediately without maintaining an in-process snapshot."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        return await self.list_bindings(subject_person_id=subject_person_id)

    async def list_bindings(
        self,
        *,
        subject_person_id: str | None = None,
    ) -> tuple[RoleBinding, ...]:
        try:
            async with self._sessions() as session:
                statement = select(RbacRoleBindingRecord)
                if subject_person_id is not None:
                    statement = statement.where(
                        RbacRoleBindingRecord.subject_person_id == subject_person_id
                    )
                records = (
                    await session.scalars(
                        statement.order_by(
                            RbacRoleBindingRecord.subject_person_id,
                            RbacRoleBindingRecord.scope,
                            RbacRoleBindingRecord.area_id,
                            RbacRoleBindingRecord.channel_id,
                            RbacRoleBindingRecord.role,
                        )
                    )
                ).all()
                return tuple(self._to_domain(record) for record in records)
        except SQLAlchemyError as exc:
            raise _database_error("load RBAC role bindings", exc) from exc

    async def grant(self, binding: RoleBinding) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        postgresql_insert(RbacRoleBindingRecord)
                        .values(
                            subject_person_id=binding.subject_person_id,
                            role=binding.role,
                            scope=binding.scope,
                            area_id=binding.area_id,
                            channel_id=binding.channel_id,
                            granted_by_person_id=binding.granted_by_person_id,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[
                                RbacRoleBindingRecord.subject_person_id,
                                RbacRoleBindingRecord.role,
                                RbacRoleBindingRecord.scope,
                                RbacRoleBindingRecord.area_id,
                                RbacRoleBindingRecord.channel_id,
                            ]
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("grant RBAC role binding", exc) from exc

    async def revoke(
        self,
        subject_person_id: str,
        role: AccessRole,
        scope: RoleBindingScope,
        *,
        area_id: str = "",
        channel_id: str = "",
    ) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        delete(RbacRoleBindingRecord).where(
                            RbacRoleBindingRecord.subject_person_id == subject_person_id,
                            RbacRoleBindingRecord.role == role,
                            RbacRoleBindingRecord.scope == scope,
                            RbacRoleBindingRecord.area_id == area_id,
                            RbacRoleBindingRecord.channel_id == channel_id,
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("revoke RBAC role binding", exc) from exc

    @staticmethod
    def _to_domain(record: RbacRoleBindingRecord) -> RoleBinding:
        return RoleBinding(
            subject_person_id=record.subject_person_id,
            role=AccessRole(record.role),
            scope=RoleBindingScope(record.scope),
            area_id=record.area_id,
            channel_id=record.channel_id,
            granted_by_person_id=record.granted_by_person_id,
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning("Failed to %s: error=%s", operation, type(error).__name__)
    return DatabaseError(f"Failed to {operation}")

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.access.models import (
    AccessPrincipal,
    AccessResource,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.repository import SqlAlchemyRoleBindingRepository
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_rbac_migration_and_fresh_repository_on_postgresql() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_rbac_test_{uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    test_engine = None
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        test_engine = create_async_engine(
            database_url,
            connect_args={"server_settings": {"search_path": schema}},
        )
        await _migrate(test_engine, "upgrade", "head")
        sessions = async_sessionmaker(test_engine, expire_on_commit=False)
        repository = SqlAlchemyRoleBindingRepository(sessions)
        service = AuthorizationService(repository)
        role_binding = RoleBinding(
            subject_person_id="person",
            role=AccessRole.ADMIN,
            scope=RoleBindingScope.AREA,
            area_id="area",
            granted_by_person_id="owner",
        )

        assert await repository.grant(role_binding) is True
        assert await repository.grant(role_binding) is False
        assert await service.allows(
            AccessPrincipal("person"),
            Permission.CHANNEL_INITIALIZE,
            AccessResource.channel("area", "channel"),
        )
        assert not await service.allows(
            AccessPrincipal("person"),
            Permission.BOT_REBOOT,
            AccessResource.global_resource(),
        )
        assert await repository.revoke(
            "person", AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"
        )
        assert not await service.allows(
            AccessPrincipal("person"),
            Permission.CHANNEL_INITIALIZE,
            AccessResource.channel("area", "channel"),
        )

        async with test_engine.connect() as connection:
            enum_names = set(
                (
                    await connection.execute(
                        text(
                            """
                            SELECT typname
                            FROM pg_type
                            JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                            WHERE pg_namespace.nspname = current_schema()
                              AND typname IN (
                                'rbac_role', 'rbac_scope', 'oopz_message_scope',
                                'oopz_outbound_message_kind', 'oopz_outbound_message_state'
                              )
                            """
                        )
                    )
                ).scalars()
            )
            diagnostics_default = await connection.scalar(
                text(
                    """
                    SELECT column_default
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'agent_runs'
                      AND column_name = 'diagnostics'
                    """
                )
            )
        assert enum_names == {
            "rbac_role",
            "rbac_scope",
            "oopz_message_scope",
            "oopz_outbound_message_kind",
            "oopz_outbound_message_state",
        }
        assert diagnostics_default == "'{}'::jsonb"

        await _migrate(test_engine, "downgrade", "20260812_23")
        async with test_engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('rbac_role_bindings')")) is None
    finally:
        if test_engine is not None:
            await test_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


async def _migrate(engine, operation: str, revision: str) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(
            lambda sync_connection: _run_alembic(sync_connection, operation, revision)
        )


def _run_alembic(connection, operation: str, revision: str) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    getattr(command, operation)(config, revision)

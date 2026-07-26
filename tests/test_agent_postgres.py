from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.agent.catalog import ReloadableProviderCatalog
from cywl_oopz.features.agent.models import (
    AgentRun,
    AgentRunLimits,
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    AgentThread,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ModelSelectionSource,
    ProviderProtocol,
)
from cywl_oopz.features.agent.repository import (
    SqlAlchemyAgentRunRepository,
    SqlAlchemyAgentThreadRepository,
    SqlAlchemyModelSelectionRepository,
    SqlAlchemyProviderCatalogRepository,
)
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.storage.models import (
    AgentRunRecord,
    ChannelSettingsRecord,
    LlmModelRecord,
    UserLlmPreferenceRecord,
)
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_agent_migration_constraints_and_repositories_on_postgresql() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_agent_test_{uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    test_engine = None
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

        test_engine = create_async_engine(
            database_url,
            connect_args={"server_settings": {"search_path": schema}},
        )
        await _migrate(test_engine, "upgrade", "20260726_01")
        async with test_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO channel_settings (
                        id, area_id, channel_id, chat_enabled, created_at, updated_at
                    )
                    VALUES (
                        :id, 'legacy-area', 'legacy-channel', true, now(), now()
                    )
                    """
                ),
                {"id": uuid4()},
            )
        await _migrate(test_engine, "upgrade", "head")
        sessions = async_sessionmaker(test_engine, expire_on_commit=False)
        async with sessions() as session:
            legacy_record = await session.scalar(
                select(ChannelSettingsRecord).where(ChannelSettingsRecord.area_id == "legacy-area")
            )
        assert legacy_record is not None
        assert legacy_record.chat_enabled is True
        assert legacy_record.default_model_id is None

        provider_id = uuid4()
        application_model_id = uuid4()
        preferred_model_id = uuid4()
        catalog_repository = SqlAlchemyProviderCatalogRepository(sessions)
        await catalog_repository.upsert_provider_bundle(
            LlmProvider(
                id=provider_id,
                alias="integration",
                display_name="Integration provider",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                base_url="https://llm.example/v1",
                api_key="database-key",
                user_selectable=True,
                enabled=True,
            ),
            (
                LlmModel(
                    id=application_model_id,
                    provider_id=provider_id,
                    alias="application",
                    remote_model_name="application-model",
                    display_name="Application model",
                    enabled=True,
                    is_provider_default=True,
                    is_application_default=True,
                    capabilities=frozenset({ModelCapability.TOOL_CALLING}),
                ),
                LlmModel(
                    id=preferred_model_id,
                    provider_id=provider_id,
                    alias="preferred",
                    remote_model_name="preferred-model",
                    display_name="Preferred model",
                    enabled=True,
                    is_provider_default=False,
                    is_application_default=False,
                    capabilities=frozenset(
                        {
                            ModelCapability.TOOL_CALLING,
                            ModelCapability.STREAMING,
                        }
                    ),
                ),
            ),
        )
        async with sessions() as session:
            async with session.begin():
                session.add(
                    UserLlmPreferenceRecord(
                        person_id="person",
                        preferred_model_id=preferred_model_id,
                    )
                )
                session.add(
                    ChannelSettingsRecord(
                        area_id="area",
                        channel_id="channel",
                        chat_enabled=True,
                        default_model_id=application_model_id,
                    )
                )

        catalog = ReloadableProviderCatalog(catalog_repository)
        await catalog.reload()
        selected = await ProviderSelectionService(
            catalog,
            SqlAlchemyModelSelectionRepository(sessions),
        ).resolve(
            ConversationKey("channel", "area", "channel", "person"),
            required_capabilities=frozenset({ModelCapability.TOOL_CALLING}),
        )

        assert selected.model.model_id == preferred_model_id
        assert selected.source is ModelSelectionSource.USER

        now = datetime.now(UTC)
        key = ConversationKey("channel", "area", "channel", "person")
        thread = AgentThread(
            id=uuid4(),
            key=key,
            selected_model_id=preferred_model_id,
            expires_at=now + timedelta(hours=1),
        )
        thread_repository = SqlAlchemyAgentThreadRepository(sessions)
        await thread_repository.add(thread)
        assert await thread_repository.get(key) == thread

        run_repository = SqlAlchemyAgentRunRepository(sessions)
        run_id = uuid4()
        running = AgentRunState(run_id).start(now)
        await run_repository.add(
            AgentRun(
                id=run_id,
                thread_id=thread.id,
                provider_id=provider_id,
                model_id=preferred_model_id,
                selection_source=ModelSelectionSource.USER,
                limits=AgentRunLimits(),
                state=running,
                heartbeat_at=now,
            )
        )
        await run_repository.finish(
            running.finish(AgentStopReason.COMPLETED, now + timedelta(seconds=1)),
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        async with sessions() as session:
            completed = await session.scalar(
                select(AgentRunRecord).where(AgentRunRecord.id == run_id)
            )
        assert completed is not None
        assert completed.status == AgentRunStatus.SUCCEEDED
        assert completed.stop_reason == AgentStopReason.COMPLETED

        stale_id = uuid4()
        stale_at = now - timedelta(hours=2)
        await run_repository.add(
            AgentRun(
                id=stale_id,
                thread_id=thread.id,
                provider_id=provider_id,
                model_id=preferred_model_id,
                selection_source=ModelSelectionSource.USER,
                limits=AgentRunLimits(),
                state=AgentRunState(stale_id).start(stale_at),
                heartbeat_at=stale_at,
            )
        )
        assert (
            await run_repository.abandon_stale(
                now - timedelta(hours=1),
                now,
            )
            == 1
        )

        with pytest.raises(IntegrityError):
            async with sessions.begin() as session:
                session.add(
                    LlmModelRecord(
                        id=uuid4(),
                        provider_id=provider_id,
                        alias="duplicate-application-default",
                        remote_model_name="duplicate",
                        display_name="Duplicate",
                        enabled=True,
                        is_provider_default=False,
                        is_application_default=True,
                        capabilities=[],
                        limits={},
                        pricing={},
                    )
                )

        await _migrate(test_engine, "downgrade", "base")
    finally:
        if test_engine is not None:
            await test_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


async def _migrate(engine, operation: str, revision: str) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(
            lambda sync_connection: _run_alembic(
                sync_connection,
                operation,
                revision,
            )
        )


def _run_alembic(connection, operation: str, revision: str) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    getattr(command, operation)(config, revision)

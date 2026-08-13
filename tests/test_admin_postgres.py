from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.admin.models import AreaChannelCatalog, ChannelKey
from cywl_oopz.features.admin.repository import SqlAlchemyChannelInitializationRepository
from cywl_oopz.storage.models import ChannelSettingsRecord, VoiceChannelSettingsRecord
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_channel_initialization_uses_defaults_and_preserves_existing_postgres_rows() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_admin_test_{uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    test_engine = None
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        test_engine = create_async_engine(
            database_url,
            connect_args={"server_settings": {"search_path": schema}},
        )
        await _migrate(test_engine)
        sessions = async_sessionmaker(test_engine, expire_on_commit=False)
        repository = SqlAlchemyChannelInitializationRepository(sessions)

        current = ChannelKey("area", "text-existing")
        assert (await repository.initialize_text_channel(current)).created is True
        assert (await repository.initialize_text_channel(current)).created is False

        async with sessions() as session:
            async with session.begin():
                text_existing = await session.scalar(
                    select(ChannelSettingsRecord).where(
                        ChannelSettingsRecord.area_id == "area",
                        ChannelSettingsRecord.channel_id == "text-existing",
                    )
                )
                assert text_existing is not None
                assert text_existing.chat_enabled is False
                text_existing.chat_enabled = True
                text_existing.enabled_agent_tools = ["custom_tool"]
                session.add(
                    VoiceChannelSettingsRecord(
                        area_id="area",
                        voice_channel_id="voice-existing",
                        enabled=True,
                        delegated_task_profile="custom_profile",
                        idle_timeout_seconds=42,
                    )
                )

        catalog = AreaChannelCatalog(
            "area",
            (current, ChannelKey("area", "text-new")),
            (
                ChannelKey("area", "voice-existing"),
                ChannelKey("area", "voice-new"),
            ),
        )
        result = await repository.initialize_area(catalog)
        assert result.text_created == 1
        assert result.text_existing == 1
        assert result.voice_created == 1
        assert result.voice_existing == 1

        repeated = await repository.initialize_area(catalog)
        assert repeated.text_created == 0
        assert repeated.text_existing == 2
        assert repeated.voice_created == 0
        assert repeated.voice_existing == 2

        async with sessions() as session:
            preserved_text = await session.scalar(
                select(ChannelSettingsRecord).where(
                    ChannelSettingsRecord.area_id == "area",
                    ChannelSettingsRecord.channel_id == "text-existing",
                )
            )
            preserved_voice = await session.get(
                VoiceChannelSettingsRecord,
                ("area", "voice-existing"),
            )
            new_voice = await session.get(
                VoiceChannelSettingsRecord,
                ("area", "voice-new"),
            )
        assert preserved_text is not None
        assert preserved_text.chat_enabled is True
        assert preserved_text.enabled_agent_tools == ["custom_tool"]
        assert preserved_voice is not None
        assert preserved_voice.enabled is True
        assert preserved_voice.delegated_task_profile == "custom_profile"
        assert preserved_voice.idle_timeout_seconds == 42
        assert new_voice is not None
        assert new_voice.enabled is False
        assert new_voice.delegated_task_profile == "voice_readonly_v1"
        assert new_voice.idle_timeout_seconds == 300
    finally:
        if test_engine is not None:
            await test_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


async def _migrate(engine) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(_run_alembic)


def _run_alembic(connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    command.upgrade(config, "head")

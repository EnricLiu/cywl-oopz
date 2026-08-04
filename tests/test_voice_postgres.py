from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.repository import (
    SqlAlchemyVoiceConfigurationRepository,
    SqlAlchemyVoiceSessionRepository,
)
from cywl_oopz.features.voice.settings import (
    PersistedVoiceSessionStatus,
    VoiceTurnRole,
)
from cywl_oopz.storage.models import (
    VoiceModelRecord,
    VoiceSessionRecord,
    VoiceTurnRecord,
)
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_voice_migration_and_repositories_on_postgresql() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_voice_test_{uuid4().hex}"
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

        async with test_engine.begin() as connection:
            defaults = (
                (
                    await connection.execute(
                        text(
                            """
                            INSERT INTO voice_providers (
                                alias, display_name, protocol, endpoint
                            ) VALUES (
                                'qwen', 'Qwen Realtime',
                                'qwen_omni_realtime_ws', 'wss://voice.example/realtime'
                            )
                            RETURNING id, credentials, config, enabled,
                                      user_selectable, created_at, updated_at
                            """
                        )
                    )
                )
                .mappings()
                .one()
            )
            provider_id = defaults["id"]
            model_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_models (
                        provider_id, alias, remote_model_name, display_name,
                        is_provider_default, is_application_default
                    ) VALUES (
                        :provider_id, 'omni', 'qwen-omni-realtime', 'Qwen Omni', true, true
                    ) RETURNING id
                    """
                ),
                {"provider_id": provider_id},
            )
            second_model_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_models (
                        provider_id, alias, remote_model_name, display_name
                    ) VALUES (
                        :provider_id, 'omni-fast', 'qwen-omni-realtime-fast',
                        'Qwen Omni Fast'
                    ) RETURNING id
                    """
                ),
                {"provider_id": provider_id},
            )
            audio_provider_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_providers (
                        alias, display_name, protocol, endpoint
                    ) VALUES (
                        'qwen-audio', 'Qwen Audio Realtime',
                        'qwen_audio_realtime_ws', 'wss://voice.example/realtime'
                    ) RETURNING id
                    """
                )
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO voice_channel_settings (area_id, voice_channel_id, enabled)
                    VALUES ('area', 'voice', true)
                    """
                )
            )
        assert defaults["credentials"] == {}
        assert defaults["config"] == {}
        assert defaults["enabled"] is True
        assert defaults["user_selectable"] is True
        assert defaults["created_at"] == defaults["updated_at"]
        assert audio_provider_id is not None

        async with test_engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE voice_providers "
                    'SET credentials = \'{"api_key": "development-secret"}\'::jsonb '
                    "WHERE id = :provider_id"
                ),
                {"provider_id": provider_id},
            )

        configuration_repository = SqlAlchemyVoiceConfigurationRepository(sessions)
        first = await configuration_repository.resolve_start_configuration(
            "person", VoiceChannelKey("area", "voice")
        )
        assert first.model.id == model_id
        assert first.model.audio_config == {}
        assert first.provider.credentials["api_key"] == "development-secret"
        assert "api_key" not in repr(first.provider)
        with pytest.raises(TypeError):
            first.model.audio_config["output_rate"] = 48_000

        await configuration_repository.set_user_voice("person", "Cherry")
        models = await configuration_repository.list_selectable_models("person")
        assert len(models) == 2
        assert next(model for model in models if model.id == model_id).selected is True
        selected = await configuration_repository.set_user_model("person", "qwen/omni-fast")
        assert selected.selected is True
        assert selected.id == second_model_id
        assert selected.selector == "qwen/omni-fast"

        async with sessions.begin() as session:
            await session.execute(
                update(VoiceModelRecord)
                .where(VoiceModelRecord.id == second_model_id)
                .values(audio_config={"output_rate": 24000})
            )
        second = await configuration_repository.resolve_start_configuration(
            "person", VoiceChannelKey("area", "voice")
        )
        assert second.voice_id == "Cherry"
        assert second.model.id == second_model_id
        assert first.model.audio_config == {}
        assert second.model.audio_config == {"output_rate": 24000}

        descriptor = VoiceSessionDescriptor(
            uuid4(),
            "person",
            VoiceChannelKey("area", "voice"),
            VoiceTextAddress("area", "text"),
        )
        history = SqlAlchemyVoiceSessionRepository(sessions)
        await history.create(descriptor, second)
        await history.mark_active(descriptor.session_id)
        await history.append_final_turn(
            descriptor.session_id,
            1,
            VoiceTurnRole.USER,
            "你好，未来。",
            provider_item_id="item-1",
            usage={"input_tokens": 3},
        )
        await history.finish(
            descriptor.session_id,
            PersistedVoiceSessionStatus.ENDED,
            "command",
            usage={"input_tokens": 3},
        )
        async with sessions() as session:
            stored_session = await session.get(VoiceSessionRecord, descriptor.session_id)
            stored_turn = await session.scalar(
                select(VoiceTurnRecord).where(VoiceTurnRecord.session_id == descriptor.session_id)
            )
        assert stored_session is not None
        assert stored_session.status is PersistedVoiceSessionStatus.ENDED
        assert stored_session.voice_id == "Cherry"
        assert stored_session.ended_at is not None
        assert stored_turn is not None
        assert stored_turn.transcript == "你好，未来。"
        assert stored_turn.usage == {"input_tokens": 3}

        stale_descriptors = [
            VoiceSessionDescriptor(
                uuid4(),
                f"stale-person-{index}",
                VoiceChannelKey("area", "voice"),
                VoiceTextAddress("area", "text"),
            )
            for index in range(3)
        ]
        for stale_descriptor in stale_descriptors:
            await history.create(stale_descriptor, second)
        await history.mark_active(stale_descriptors[1].session_id)
        await history.mark_active(stale_descriptors[2].session_id)
        await history.mark_recovering(stale_descriptors[2].session_id)

        recovered_at = datetime.now(UTC)
        assert await history.recover_stale(recovered_at) == 3
        assert await history.recover_stale(datetime.now(UTC)) == 0
        async with sessions() as session:
            recovered_sessions = (
                (
                    await session.execute(
                        select(VoiceSessionRecord).where(
                            VoiceSessionRecord.id.in_(
                                tuple(item.session_id for item in stale_descriptors)
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )
            original_session = await session.get(VoiceSessionRecord, descriptor.session_id)
        assert len(recovered_sessions) == 3
        assert all(
            item.status is PersistedVoiceSessionStatus.FAILED
            and item.ended_at == recovered_at
            and item.stop_reason == "process_restarted"
            for item in recovered_sessions
        )
        assert original_session is not None
        assert original_session.status is PersistedVoiceSessionStatus.ENDED
        assert original_session.stop_reason == "command"

        async with test_engine.connect() as connection:
            trigger_count = await connection.scalar(
                text(
                    """
                    SELECT count(*) FROM pg_trigger
                    JOIN pg_class ON pg_class.oid = pg_trigger.tgrelid
                    JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
                    WHERE pg_namespace.nspname = current_schema()
                      AND tgname LIKE 'trg_voice_%_set_updated_at'
                      AND NOT pg_trigger.tgisinternal
                    """
                )
            )
        assert trigger_count == 5

        async with test_engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM voice_providers WHERE id = :provider_id"),
                {"provider_id": audio_provider_id},
            )

        await _migrate(test_engine, "downgrade", "20260730_18")
        async with test_engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('voice_providers')")) is None
            assert (
                await connection.scalar(
                    text(
                        """
                            SELECT EXISTS (
                                SELECT 1
                                FROM pg_type
                                JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                                WHERE pg_type.typname = 'voice_model_mode'
                                  AND pg_namespace.nspname = current_schema()
                            )
                            """
                    )
                )
                is False
            )
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

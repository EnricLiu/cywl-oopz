from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.agent.catalog import ReloadableProviderCatalog
from cywl_oopz.features.agent.memory import MemoryItem
from cywl_oopz.features.agent.memory_repository import SqlAlchemyMemoryRepository
from cywl_oopz.features.agent.models import (
    AgentMessage,
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
    SqlAlchemyAgentMessageRepository,
    SqlAlchemyAgentRunRepository,
    SqlAlchemyAgentThreadRepository,
    SqlAlchemyModelSelectionRepository,
    SqlAlchemyProviderCatalogRepository,
    SqlAlchemyToolExecutionRepository,
)
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.agent.tools.models import (
    ToolEffect,
    ToolExecution,
    ToolExecutionStatus,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.storage.models import (
    AgentRunRecord,
    ChannelSettingsRecord,
    LlmModelRecord,
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
        assert legacy_record.enabled_agent_tools == [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "skip_music",
            "pause_music",
            "resume_music",
            "search_web",
        ]

        async with test_engine.begin() as connection:
            defaulted = (
                (
                    await connection.execute(
                        text(
                            """
                        INSERT INTO channel_settings (area_id, channel_id)
                        VALUES ('default-area', 'default-channel')
                        RETURNING
                            id,
                            chat_enabled,
                            enabled_agent_tools,
                            created_at,
                            updated_at
                        """
                        )
                    )
                )
                .mappings()
                .one()
            )
        assert defaulted["id"] is not None
        assert defaulted["chat_enabled"] is False
        assert defaulted["enabled_agent_tools"] == [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "skip_music",
            "pause_music",
            "resume_music",
            "search_web",
        ]
        assert defaulted["created_at"] == defaulted["updated_at"]

        async with test_engine.begin() as connection:
            triggered_updated_at = await connection.scalar(
                text(
                    """
                    UPDATE channel_settings
                    SET chat_enabled = true
                    WHERE id = :id
                    RETURNING updated_at
                    """
                ),
                {"id": defaulted["id"]},
            )
            enum_columns = (
                (
                    await connection.execute(
                        text(
                            """
                        SELECT table_name, column_name, udt_name
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND (table_name, column_name) IN (
                            ('agent_runs', 'status'),
                            ('agent_runs', 'stop_reason'),
                            ('agent_runs', 'selection_source'),
                            ('agent_tool_executions', 'effect'),
                            ('agent_tool_executions', 'status')
                          )
                        """
                        )
                    )
                )
                .mappings()
                .all()
            )
        assert triggered_updated_at > defaulted["updated_at"]
        assert {
            (row["table_name"], row["column_name"]): row["udt_name"] for row in enum_columns
        } == {
            ("agent_runs", "status"): "agent_run_status",
            ("agent_runs", "stop_reason"): "agent_stop_reason",
            ("agent_runs", "selection_source"): "model_selection_source",
            ("agent_tool_executions", "effect"): "tool_effect",
            ("agent_tool_executions", "status"): "tool_execution_status",
        }

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
                    ChannelSettingsRecord(
                        area_id="area",
                        channel_id="channel",
                        chat_enabled=True,
                        default_model_id=application_model_id,
                    )
                )

        catalog = ReloadableProviderCatalog(catalog_repository)
        await catalog.reload()
        selection_repository = SqlAlchemyModelSelectionRepository(sessions)
        await selection_repository.set_user_model("person", preferred_model_id)
        selected = await ProviderSelectionService(
            catalog,
            selection_repository,
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
        tool_execution_repository = SqlAlchemyToolExecutionRepository(sessions)
        tool_execution = ToolExecution(
            id=uuid4(),
            run_id=run_id,
            call_id="call-status",
            tool_name="get_agent_status",
            tool_version="1",
            effect=ToolEffect.READ,
            status=ToolExecutionStatus.STARTED,
            idempotency_key=f"{run_id}:call-status",
            input_payload={},
            output_payload=None,
            error_code="",
            started_at=now,
        )
        first_claim = await tool_execution_repository.claim(tool_execution)
        duplicate_claim = await tool_execution_repository.claim(tool_execution)
        semantic_duplicate = await tool_execution_repository.claim(
            replace(
                tool_execution,
                id=uuid4(),
                call_id="call-status-repeated",
            )
        )
        assert first_claim.created is True
        assert duplicate_claim.created is False
        assert semantic_duplicate.created is False
        assert semantic_duplicate.execution.call_id == "call-status"
        completed_tool = await tool_execution_repository.finish(
            run_id,
            "call-status",
            ToolExecutionStatus.SUCCEEDED,
            output={"mode": "agent"},
            error_code="",
        )
        assert completed_tool.output_payload == {"mode": "agent"}
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
        with pytest.raises(DBAPIError):
            async with test_engine.begin() as connection:
                await connection.execute(
                    text("UPDATE agent_runs SET status = 'unknown' WHERE id = :id"),
                    {"id": run_id},
                )

        message_repository = SqlAlchemyAgentMessageRepository(sessions)
        await message_repository.append(
            thread.id,
            run_id,
            (
                AgentMessage("user", "text", {"text": "question"}),
                AgentMessage(
                    "assistant",
                    "text",
                    {"text": "answer"},
                    input_tokens=10,
                    output_tokens=5,
                ),
            ),
        )
        assert await message_repository.count(thread.id) == 2
        loaded_messages = await message_repository.load(thread.id, limit=10)
        assert [message.content["text"] for message in loaded_messages] == [
            "question",
            "answer",
        ]
        assert [message.sequence for message in loaded_messages] == [1, 2]
        assert await thread_repository.save_summary(
            thread.id,
            "question and answer",
            2,
            expected_version=thread.version,
        )
        summarized_thread = await thread_repository.get(key)
        assert summarized_thread is not None
        assert summarized_thread.summary == "question and answer"
        assert summarized_thread.summary_through_sequence == 2
        assert summarized_thread.summary_version == 1
        assert (
            await message_repository.load(
                thread.id,
                limit=10,
                after_sequence=2,
            )
            == ()
        )

        memory_repository = SqlAlchemyMemoryRepository(sessions)
        memory_id = uuid4()
        memory_item = MemoryItem(
            id=memory_id,
            owner_person_id="person",
            namespace="explicit",
            content={"text": "likes jazz"},
            source_thread_id=thread.id,
            source_message_sequence=2,
            created_at=now,
            updated_at=now,
            last_used_at=None,
            expires_at=now + timedelta(days=30),
        )
        await memory_repository.add(memory_item)
        await memory_repository.set_preference("person", False)
        assert await memory_repository.preference("person") is False
        assert await memory_repository.count_active("person", now) == 1
        assert await memory_repository.count_active("other", now) == 0
        assert await memory_repository.list_active("person", now, limit=10) == (memory_item,)
        assert await memory_repository.delete("other", memory_id) is False
        await memory_repository.touch("person", (memory_id,), now)

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
        await message_repository.append(
            thread.id,
            stale_id,
            (AgentMessage("user", "text", {"text": "stale question"}),),
        )
        assert (
            await run_repository.abandon_stale(
                now - timedelta(hours=1),
                now,
            )
            == 1
        )
        assert await message_repository.count(thread.id) == 2
        visible_after_abandon = await message_repository.load(thread.id, limit=10)
        assert [message.content["text"] for message in visible_after_abandon] == [
            "question",
            "answer",
        ]

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

        await thread_repository.delete(key)
        memory_after_thread_delete = await memory_repository.list_active(
            "person",
            now,
            limit=10,
        )
        assert len(memory_after_thread_delete) == 1
        assert memory_after_thread_delete[0].source_thread_id is None

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

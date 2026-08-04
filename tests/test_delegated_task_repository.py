from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.agent.delegation.models import (
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskStatus,
    DelegatedTaskSubmission,
    TaskListQuery,
    TaskRef,
)
from cywl_oopz.features.agent.delegation.repository import (
    SqlAlchemyDelegatedTaskRepository,
)
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_delegated_task_migration_repository_and_state_machine_on_postgresql() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_task_test_{uuid4().hex}"
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
            agent_provider_id = await connection.scalar(
                text(
                    """
                    INSERT INTO llm_providers (
                        alias, display_name, protocol, base_url, api_key
                    ) VALUES (
                        'agent', 'Agent', 'openai_compatible',
                        'https://agent.invalid/v1', 'development-secret'
                    ) RETURNING id
                    """
                )
            )
            agent_model_id = await connection.scalar(
                text(
                    """
                    INSERT INTO llm_models (
                        provider_id, alias, remote_model_name, display_name,
                        is_provider_default, is_application_default
                    ) VALUES (
                        :provider_id, 'default', 'agent-model', 'Agent Model', true, true
                    ) RETURNING id
                    """
                ),
                {"provider_id": agent_provider_id},
            )
            voice_provider_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_providers (
                        alias, display_name, protocol, endpoint
                    ) VALUES (
                        'qwen', 'Qwen', 'qwen_omni_realtime_ws',
                        'wss://voice.invalid/realtime'
                    ) RETURNING id
                    """
                )
            )
            voice_model_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_models (
                        provider_id, alias, remote_model_name, display_name,
                        is_provider_default, is_application_default
                    ) VALUES (
                        :provider_id, 'omni', 'qwen-omni', 'Qwen Omni', true, true
                    ) RETURNING id
                    """
                ),
                {"provider_id": voice_provider_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO voice_channel_settings (
                        area_id, voice_channel_id, enabled
                    ) VALUES ('area', 'voice', true)
                    """
                )
            )
            voice_session_id = await connection.scalar(
                text(
                    """
                    INSERT INTO voice_sessions (
                        owner_person_id, area_id, voice_channel_id,
                        text_channel_id, model_id, duplex_mode, status
                    ) VALUES (
                        'owner', 'area', 'voice', 'text', :model_id, 'full', 'active'
                    ) RETURNING id
                    """
                ),
                {"model_id": voice_model_id},
            )

        repository = SqlAlchemyDelegatedTaskRepository(sessions)
        policy = await repository.resolve_submission_policy(voice_session_id, "owner")
        assert policy.profile == "voice_readonly_v1"
        assert policy.agent_model_id == agent_model_id

        duplicate_request = _submission(voice_session_id, agent_model_id, "call-duplicate")
        duplicate_tasks = await asyncio.gather(
            repository.submit(duplicate_request),
            repository.submit(duplicate_request),
        )
        assert duplicate_tasks[0].id == duplicate_tasks[1].id
        assert duplicate_tasks[0].alias == "T1"

        distinct = await asyncio.gather(
            repository.submit(_submission(voice_session_id, agent_model_id, "call-2")),
            repository.submit(_submission(voice_session_id, agent_model_id, "call-3")),
        )
        assert {task.alias for task in distinct} == {"T2", "T3"}
        assert len({task.id for task in distinct}) == 2

        assert (
            await repository.get_for_owner(
                TaskRef.parse("T1", origin_voice_session_id=voice_session_id),
                "another-owner",
            )
            is None
        )
        listed = await repository.list_for_owner(
            "owner",
            TaskListQuery(origin_voice_session_id=voice_session_id),
        )
        assert [task.alias for task in listed] == ["T3", "T2", "T1"]

        claimed = await repository.claim_next(
            "worker-1",
            frozenset({DelegatedTaskLane.READ_PARALLEL}),
        )
        assert claimed is not None
        assert claimed.status is DelegatedTaskStatus.RUNNING
        assert await repository.update_progress(
            claimed.id,
            "worker-1",
            "searching",
            "正在搜索公开网页",
        )
        assert await repository.complete(
            claimed.id,
            "worker-1",
            "找到三条结果",
            "完整结果",
        )
        completed = await repository.get_for_owner(TaskRef(task_id=claimed.id), "owner")
        assert completed is not None
        assert completed.status is DelegatedTaskStatus.SUCCEEDED
        assert completed.notification_state is DelegatedTaskNotificationState.PENDING

        notifications = await repository.claim_notifications(voice_session_id, 5)
        assert [task.id for task in notifications] == [claimed.id]
        assert notifications[0].notification_state is DelegatedTaskNotificationState.CLAIMED
        await repository.mark_presented((claimed.id,))
        presented = await repository.get_for_owner(TaskRef(task_id=claimed.id), "owner")
        assert presented is not None
        assert presented.notification_state is DelegatedTaskNotificationState.PRESENTED

        queued = next(task for task in distinct if task.id != claimed.id)
        cancelled = await repository.request_cancel(queued.id, "owner")
        assert cancelled.cancel_requested is True
        assert cancelled.task is not None
        assert cancelled.task.status is DelegatedTaskStatus.CANCELLED

        running = await repository.claim_next(
            "worker-2",
            frozenset({DelegatedTaskLane.READ_PARALLEL}),
        )
        assert running is not None
        retry_at = datetime.now(UTC) + timedelta(seconds=5)
        assert await repository.mark_waiting_retry(
            running.id,
            "worker-2",
            retry_at,
            "provider_unavailable",
        )
        assert (
            await repository.claim_next(
                "worker-3",
                frozenset({DelegatedTaskLane.READ_PARALLEL}),
            )
            is None
        )

        async with test_engine.connect() as connection:
            defaults = (
                (
                    await connection.execute(
                        text(
                            """
                            SELECT status, lane, notification_state, retry_count,
                                   allowed_tool_names, created_at, updated_at
                            FROM delegated_agent_tasks
                            WHERE id = :task_id
                            """
                        ),
                        {"task_id": running.id},
                    )
                )
                .mappings()
                .one()
            )
        assert defaults["status"] == "waiting_retry"
        assert defaults["lane"] == "read_parallel"
        assert defaults["retry_count"] == 1
        assert defaults["allowed_tool_names"] == ["search_web", "read_web_page"]
        assert defaults["updated_at"] >= defaults["created_at"]

        recovery = await repository.recover_stale(datetime.now(UTC))
        assert recovery.requeued == 0
        assert recovery.cancelled == 0
        assert recovery.interrupted == 0

        await _migrate(test_engine, "downgrade", "20260804_19")
        async with test_engine.connect() as connection:
            assert (
                await connection.scalar(text("SELECT to_regclass('delegated_agent_tasks')")) is None
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_type "
                        "WHERE typname='delegated_task_status')"
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


def _submission(session_id, model_id, call_id: str) -> DelegatedTaskSubmission:
    return DelegatedTaskSubmission(
        owner_person_id="owner",
        area_id="area",
        text_channel_id="text",
        voice_channel_id="voice",
        origin_voice_session_id=session_id,
        provider_call_id=call_id,
        objective=f"查询 {call_id}",
        result_style=DelegatedResultStyle.BRIEF,
        lane=DelegatedTaskLane.READ_PARALLEL,
        conflict_key="",
        agent_model_id=model_id,
        allowed_tool_names=("search_web", "read_web_page"),
    )


async def _migrate(engine, operation: str, revision: str) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(
            lambda sync_connection: _run_alembic(sync_connection, operation, revision)
        )


def _run_alembic(connection, operation: str, revision: str) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    getattr(command, operation)(config, revision)

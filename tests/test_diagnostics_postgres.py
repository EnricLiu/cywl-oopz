from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.core.lifecycle import (
    AgentRunStatus,
    AgentStopReason,
    ModelSelectionSource,
    ToolEffect,
    ToolExecutionStatus,
)
from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)
from cywl_oopz.features.admin.outbound_repository import (
    SqlAlchemyAgentDiagnosticRepository,
    SqlAlchemyOutboundMessageRepository,
)
from cywl_oopz.storage.models import (
    AgentMessageRecord,
    AgentRunRecord,
    AgentThreadRecord,
    AgentToolExecutionRecord,
    LlmModelRecord,
    LlmProviderRecord,
)
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_outbound_agent_receipt_bind_finalize_and_diagnostic_read_on_postgres() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_diagnostic_test_{uuid4().hex}"
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
        now = datetime.now(UTC)
        provider_id = uuid4()
        model_id = uuid4()
        thread_id = uuid4()
        run_id = uuid4()
        async with sessions() as session:
            async with session.begin():
                session.add(
                    LlmProviderRecord(
                        id=provider_id,
                        alias="provider",
                        display_name="Provider",
                        protocol="openai_chat_compatible",
                        base_url="https://provider.example",
                        api_key="development-key",
                    )
                )
                await session.flush()
                session.add(
                    LlmModelRecord(
                        id=model_id,
                        provider_id=provider_id,
                        alias="model",
                        remote_model_name="remote-model",
                        display_name="Model",
                    )
                )
                await session.flush()
                session.add(
                    AgentThreadRecord(
                        id=thread_id,
                        scope="channel",
                        area_id="area",
                        channel_id="channel",
                        person_id="person",
                        expires_at=now + timedelta(hours=1),
                    )
                )
                await session.flush()
                session.add(
                    AgentRunRecord(
                        id=run_id,
                        thread_id=thread_id,
                        status=AgentRunStatus.SUCCEEDED,
                        stop_reason=AgentStopReason.COMPLETED,
                        provider_id=provider_id,
                        model_id=model_id,
                        selection_source=ModelSelectionSource.USER,
                        limits={"max_tool_calls": 8},
                        usage={"input_tokens": 10, "output_tokens": 5, "tool_calls": 1},
                        diagnostics={"provider_retries": 1},
                        started_at=now,
                        finished_at=now + timedelta(seconds=1),
                        heartbeat_at=now + timedelta(seconds=1),
                    )
                )
                await session.flush()
                session.add(
                    AgentMessageRecord(
                        thread_id=thread_id,
                        run_id=run_id,
                        sequence=1,
                        role="assistant",
                        kind="text",
                        content={"text": "完整答案"},
                    )
                )
                session.add(
                    AgentToolExecutionRecord(
                        run_id=run_id,
                        tool_call_id="call-1",
                        tool_name="search_web",
                        tool_version="1",
                        effect=ToolEffect.READ,
                        status=ToolExecutionStatus.SUCCEEDED,
                        idempotency_key="key",
                        input_payload={"query": "初音未来"},
                        output_payload={"results": []},
                        started_at=now,
                        finished_at=now + timedelta(milliseconds=500),
                    )
                )

        address = OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel")
        writes = SqlAlchemyOutboundMessageRepository(sessions)
        assert await writes.create(
            OutboundMessageReceipt(
                "message",
                "123",
                OutboundMessageKind.AGENT_RESPONSE,
                OutboundMessageState.ACTIVE,
                address,
                in_reply_to_message_id="source",
                owner_person_id="person",
            )
        )
        assert await writes.bind_agent_run("message", run_id)
        assert await writes.update_state(
            "message",
            OutboundMessageState.FINAL,
            diagnostic_snapshot={"phase": "succeeded", "final_text": "完整答案"},
        )

        reads = SqlAlchemyAgentDiagnosticRepository(sessions)
        result = await reads.get_by_outbound_message("message", address)
        assert result is not None
        assert result.run_id == run_id
        assert result.provider_alias == "provider"
        assert result.model_alias == "model"
        assert result.assistant_text == "完整答案"
        assert result.tools[0].name == "search_web"
        assert result.receipt.state is OutboundMessageState.FINAL
        assert (
            await reads.get_by_outbound_message(
                "message",
                OopzMessageAddress(OopzMessageScope.CHANNEL, "other-area", "channel"),
            )
            is None
        )

        async with sessions() as session:
            async with session.begin():
                thread_record = await session.get(AgentThreadRecord, thread_id)
                assert thread_record is not None
                await session.delete(thread_record)
        expired = await reads.get_by_outbound_message("message", address)
        assert expired is not None
        assert expired.run_id is None
        assert expired.receipt.diagnostic_snapshot["final_text"] == "完整答案"
        exact = await writes.get_by_message("message", address)
        assert exact is not None
        assert exact.state is OutboundMessageState.FINAL
        assert await writes.mark_recalled("message")
        assert not await writes.mark_recalled("message")
        recalled = await writes.get_by_message("message", address)
        assert recalled is not None
        assert recalled.state is OutboundMessageState.RECALLED
        async with sessions() as session:
            recalled_at = await session.scalar(
                text("SELECT recalled_at FROM oopz_outbound_messages WHERE message_id = 'message'")
            )
        assert recalled_at is not None
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

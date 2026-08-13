"""PostgreSQL persistence for Bot-owned OOPZ outbound messages."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import (
    AgentMessageRecord,
    AgentRunRecord,
    AgentToolExecutionRecord,
    LlmModelRecord,
    LlmProviderRecord,
    OopzOutboundMessageRecord,
)

from .models import (
    AgentDiagnosticTool,
    AgentResponseDiagnostic,
    OopzMessageAddress,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)

logger = logging.getLogger(__name__)


class SqlAlchemyOutboundMessageRepository:
    """Persist receipts in isolated short transactions without blocking message delivery."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, receipt: OutboundMessageReceipt) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        postgresql_insert(OopzOutboundMessageRecord)
                        .values(
                            message_id=receipt.message_id,
                            message_timestamp=receipt.message_timestamp,
                            kind=receipt.kind,
                            state=receipt.state,
                            scope=receipt.address.scope,
                            area_id=receipt.address.area_id,
                            channel_id=receipt.address.channel_id,
                            target_person_id=receipt.address.target_person_id,
                            in_reply_to_message_id=receipt.in_reply_to_message_id,
                            owner_person_id=receipt.owner_person_id,
                            agent_run_id=receipt.agent_run_id,
                            diagnostic_snapshot=dict(receipt.diagnostic_snapshot),
                        )
                        .on_conflict_do_nothing(
                            index_elements=[OopzOutboundMessageRecord.message_id]
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("create outbound message receipt", exc) from exc

    async def bind_agent_run(self, message_id: str, run_id: UUID) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(OopzOutboundMessageRecord)
                        .where(
                            OopzOutboundMessageRecord.message_id == message_id,
                            OopzOutboundMessageRecord.agent_run_id.is_(None),
                        )
                        .values(agent_run_id=run_id)
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("bind outbound Agent run", exc) from exc

    async def promote_agent_response(
        self,
        message_id: str,
        run_id: UUID | None,
        diagnostic_snapshot: dict[str, object],
    ) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(OopzOutboundMessageRecord)
                        .where(OopzOutboundMessageRecord.message_id == message_id)
                        .values(
                            kind=OutboundMessageKind.AGENT_RESPONSE,
                            state=OutboundMessageState.FINAL,
                            agent_run_id=run_id,
                            diagnostic_snapshot=diagnostic_snapshot,
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("promote outbound Agent response", exc) from exc

    async def update_state(
        self,
        message_id: str,
        state: OutboundMessageState,
        *,
        diagnostic_snapshot: dict[str, object] | None = None,
    ) -> bool:
        values: dict[str, object] = {"state": state}
        if diagnostic_snapshot is not None:
            values["diagnostic_snapshot"] = diagnostic_snapshot
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(OopzOutboundMessageRecord)
                        .where(OopzOutboundMessageRecord.message_id == message_id)
                        .values(**values)
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("update outbound message state", exc) from exc

    async def get_by_message(
        self,
        message_id: str,
        address: OopzMessageAddress,
    ) -> OutboundMessageReceipt | None:
        try:
            async with self._sessions() as session:
                record = await session.scalar(
                    select(OopzOutboundMessageRecord).where(
                        OopzOutboundMessageRecord.message_id == message_id,
                        OopzOutboundMessageRecord.scope == address.scope,
                        OopzOutboundMessageRecord.area_id == address.area_id,
                        OopzOutboundMessageRecord.channel_id == address.channel_id,
                        OopzOutboundMessageRecord.target_person_id == address.target_person_id,
                    )
                )
                return self._receipt(record) if record is not None else None
        except SQLAlchemyError as exc:
            raise _database_error("load outbound message receipt", exc) from exc

    async def mark_recalled(self, message_id: str) -> bool:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    result = await session.execute(
                        update(OopzOutboundMessageRecord)
                        .where(
                            OopzOutboundMessageRecord.message_id == message_id,
                            OopzOutboundMessageRecord.state != OutboundMessageState.RECALLED,
                        )
                        .values(
                            state=OutboundMessageState.RECALLED,
                            recalled_at=datetime.now(UTC),
                        )
                    )
                    return result.rowcount == 1
        except SQLAlchemyError as exc:
            raise _database_error("mark outbound message recalled", exc) from exc

    @staticmethod
    def _receipt(record: OopzOutboundMessageRecord) -> OutboundMessageReceipt:
        return _outbound_receipt(record)


class SqlAlchemyAgentDiagnosticRepository:
    """Read only the rows belonging to one exact tracked Agent response."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_by_outbound_message(
        self,
        message_id: str,
        address: OopzMessageAddress,
    ) -> AgentResponseDiagnostic | None:
        try:
            async with self._sessions() as session:
                receipt_record = await session.scalar(
                    select(OopzOutboundMessageRecord).where(
                        OopzOutboundMessageRecord.message_id == message_id,
                        OopzOutboundMessageRecord.kind == OutboundMessageKind.AGENT_RESPONSE,
                        OopzOutboundMessageRecord.scope == address.scope,
                        OopzOutboundMessageRecord.area_id == address.area_id,
                        OopzOutboundMessageRecord.channel_id == address.channel_id,
                        OopzOutboundMessageRecord.target_person_id == address.target_person_id,
                    )
                )
                if receipt_record is None:
                    return None
                receipt = self._receipt(receipt_record)
                if receipt_record.agent_run_id is None:
                    return AgentResponseDiagnostic(receipt=receipt)
                row = (
                    await session.execute(
                        select(AgentRunRecord, LlmProviderRecord, LlmModelRecord)
                        .join(
                            LlmProviderRecord,
                            LlmProviderRecord.id == AgentRunRecord.provider_id,
                        )
                        .join(LlmModelRecord, LlmModelRecord.id == AgentRunRecord.model_id)
                        .where(AgentRunRecord.id == receipt_record.agent_run_id)
                    )
                ).one_or_none()
                if row is None:
                    return AgentResponseDiagnostic(
                        receipt=receipt,
                        run_id=receipt_record.agent_run_id,
                    )
                run, provider, model = row
                assistant_content = await session.scalar(
                    select(AgentMessageRecord.content)
                    .where(
                        AgentMessageRecord.run_id == run.id,
                        AgentMessageRecord.role == "assistant",
                        AgentMessageRecord.kind == "text",
                    )
                    .order_by(AgentMessageRecord.sequence.desc())
                    .limit(1)
                )
                tool_records = (
                    await session.scalars(
                        select(AgentToolExecutionRecord)
                        .where(AgentToolExecutionRecord.run_id == run.id)
                        .order_by(
                            AgentToolExecutionRecord.started_at,
                            AgentToolExecutionRecord.id,
                        )
                    )
                ).all()
                return AgentResponseDiagnostic(
                    receipt=receipt,
                    run_id=run.id,
                    thread_id=run.thread_id,
                    status=_value(run.status),
                    stop_reason=_value(run.stop_reason),
                    error_code=run.error_code,
                    provider_alias=provider.alias,
                    model_alias=model.alias,
                    selection_source=_value(run.selection_source),
                    limits=_mapping(run.limits),
                    usage=_mapping(run.usage),
                    run_diagnostics=_mapping(run.diagnostics),
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    assistant_text=_assistant_text(assistant_content),
                    tools=tuple(self._tool(record) for record in tool_records),
                )
        except SQLAlchemyError as exc:
            raise _database_error("load Agent response diagnostic", exc) from exc

    @staticmethod
    def _receipt(record: OopzOutboundMessageRecord) -> OutboundMessageReceipt:
        return _outbound_receipt(record)

    @staticmethod
    def _tool(record: AgentToolExecutionRecord) -> AgentDiagnosticTool:
        return AgentDiagnosticTool(
            call_id=record.tool_call_id,
            name=record.tool_name,
            version=record.tool_version,
            effect=_value(record.effect),
            status=_value(record.status),
            input_payload=_mapping(record.input_payload),
            output_payload=(
                _mapping(record.output_payload) if record.output_payload is not None else None
            ),
            error_code=record.error_code,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning("Failed to %s: error=%s", operation, type(error).__name__)
    return DatabaseError(f"Failed to {operation}")


def _outbound_receipt(record: OopzOutboundMessageRecord) -> OutboundMessageReceipt:
    return OutboundMessageReceipt(
        message_id=record.message_id,
        message_timestamp=record.message_timestamp,
        kind=record.kind,
        state=record.state,
        address=OopzMessageAddress(
            record.scope,
            record.area_id,
            record.channel_id,
            record.target_person_id,
        ),
        in_reply_to_message_id=record.in_reply_to_message_id,
        owner_person_id=record.owner_person_id,
        agent_run_id=record.agent_run_id,
        diagnostic_snapshot=_mapping(record.diagnostic_snapshot),
    )


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _assistant_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    text = value.get("text", "")
    return text if isinstance(text, str) else ""


def _value(value: object) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))

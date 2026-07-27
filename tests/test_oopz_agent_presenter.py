from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from oopz_sdk.exceptions import OopzConnectionError

from cywl_oopz.features.chat.models import ChatResponse
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.integrations.oopz.agent_presenter import (
    OopzAgentLoopMessage,
    OopzAgentPresenterFactory,
)
from cywl_oopz.integrations.oopz.editable_messages import (
    EditableMessageRef,
    MessageAddress,
)
from cywl_oopz.integrations.oopz.message_renderer import OopzMessageRenderer


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0)
        await asyncio.sleep(0)


class FakeGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.edits: list[str] = []
        self.failures: list[Exception | None] = []
        self.active_edits = 0
        self.max_active_edits = 0
        self.edit_started: asyncio.Event | None = None
        self.release_edit: asyncio.Event | None = None

    async def create_reply(
        self,
        address: MessageAddress,
        text: str,
    ) -> EditableMessageRef:
        self.created.append(text)
        return EditableMessageRef(
            message_id=f"message-{len(self.created)}",
            timestamp="123",
            scope=address.scope,
            area_id=address.area_id,
            channel_id=address.channel_id,
            target_person_id=address.target_person_id,
            reference_message_id=address.reference_message_id,
        )

    async def replace(self, message: EditableMessageRef, text: str) -> None:
        del message
        self.active_edits += 1
        self.max_active_edits = max(self.max_active_edits, self.active_edits)
        try:
            if self.edit_started is not None:
                self.edit_started.set()
            if self.release_edit is not None:
                await self.release_edit.wait()
            failure = self.failures.pop(0) if self.failures else None
            if failure is not None:
                raise failure
            self.edits.append(text)
        finally:
            self.active_edits -= 1


def address() -> MessageAddress:
    return MessageAddress("channel", "area", "channel", "", "source")


async def opened_session(
    gateway: FakeGateway,
) -> OopzAgentLoopMessage:
    clock = FakeClock()
    session = OopzAgentLoopMessage(
        gateway,  # type: ignore[arg-type]
        address(),
        OopzMessageRenderer(),
        edit_interval_seconds=0.8,
        clock=clock,
        sleep=clock.sleep,
    )
    await session.open()
    return session


@pytest.mark.asyncio
async def test_many_text_deltas_coalesce_into_one_serial_terminal_edit() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)

    await session.emit(ConversationProgressEvent(ProgressKind.ACCEPTED))
    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_RESET))
    for _ in range(100):
        await session.emit(ConversationProgressEvent(ProgressKind.TEXT_DELTA, text="字"))
    await session.complete(ChatResponse("最终回答", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 1
    assert 1 <= len(gateway.edits) <= 3
    assert gateway.edits[-1] == "🎵 **CYWL**\n最终回答"
    assert gateway.max_active_edits == 1
    assert session.state.terminal is True


@pytest.mark.asyncio
async def test_revision_arriving_during_edit_flushes_the_latest_terminal_snapshot() -> None:
    gateway = FakeGateway()
    gateway.edit_started = asyncio.Event()
    gateway.release_edit = asyncio.Event()
    session = await opened_session(gateway)

    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await asyncio.wait_for(gateway.edit_started.wait(), timeout=1)
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_RESET))
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_DELTA, text="新回答"))
    finish = asyncio.create_task(session.complete(ChatResponse("最终新回答", "provider/model")))
    gateway.release_edit.set()
    await asyncio.wait_for(finish, timeout=1)
    await session.aclose()

    assert len(gateway.edits) == 2
    assert gateway.edits[-1].endswith("最终新回答")
    assert gateway.max_active_edits == 1


@pytest.mark.asyncio
async def test_retryable_edit_keeps_latest_state_and_does_not_create_fallback() -> None:
    gateway = FakeGateway()
    gateway.failures = [OopzConnectionError("temporary"), None]
    session = await opened_session(gateway)

    await session.complete(ChatResponse("重试后成功", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 1
    assert gateway.edits[-1].endswith("重试后成功")


@pytest.mark.asyncio
async def test_deterministic_terminal_edit_failure_creates_exactly_one_fallback() -> None:
    gateway = FakeGateway()
    gateway.failures = [ValueError("unsupported edit")]
    session = await opened_session(gateway)

    await session.complete(ChatResponse("最终回答", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 2
    assert gateway.created[-1].endswith("最终回答")
    assert gateway.edits == []


@pytest.mark.asyncio
async def test_factory_degrades_to_noop_when_placeholder_creation_fails() -> None:
    class BrokenGateway(FakeGateway):
        async def create_reply(
            self,
            address: MessageAddress,
            text: str,
        ) -> EditableMessageRef:
            del address, text
            raise OopzConnectionError("offline")

    factory = OopzAgentPresenterFactory(
        BrokenGateway(),  # type: ignore[arg-type]
        OopzMessageRenderer(),
        enabled=True,
        edit_interval_seconds=0.8,
    )
    message = SimpleNamespace(
        area="area",
        channel="channel",
        sender_id="person",
        message_id="source",
    )
    context = SimpleNamespace(event=SimpleNamespace(message=message, is_private=False))

    session = await factory.open(context)

    assert session.owns_message is False
    await session.complete(ChatResponse("normal fallback", "provider/model"))
    await session.aclose()

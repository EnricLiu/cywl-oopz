from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cywl_oopz.features.voice.models import VoiceSessionState, VoiceSessionStatus
from cywl_oopz.integrations.oopz.editable_messages import EditableMessageRef, MessageAddress
from cywl_oopz.integrations.oopz.voice_status import (
    OopzVoiceStatusMessage,
    OopzVoiceStatusRenderer,
)


class FakeEditableGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.replaced: list[str] = []
        self.replace_error: Exception | None = None
        self.fallback_error: Exception | None = None
        self.fallback_started: asyncio.Event | None = None
        self.fallback_release: asyncio.Event | None = None
        self.replace_started: asyncio.Event | None = None
        self.replace_release: asyncio.Event | None = None

    async def create_reply(self, address, text):
        del address
        if self.created and self.fallback_started is not None:
            self.fallback_started.set()
        if self.created and self.fallback_release is not None:
            await self.fallback_release.wait()
        if self.created and self.fallback_error is not None:
            raise self.fallback_error
        self.created.append(text)
        return EditableMessageRef("message", "timestamp", "channel", "area", "text", "", "ref")

    async def replace(self, message, text):
        del message
        if self.replace_started is not None:
            self.replace_started.set()
        if self.replace_release is not None:
            await self.replace_release.wait()
        if self.replace_error is not None:
            raise self.replace_error
        self.replaced.append(text)


def address() -> MessageAddress:
    return MessageAddress("channel", "area", "text", "", "ref")


def status(
    state: VoiceSessionState,
    *,
    active: bool = True,
    elapsed: float = 12,
    metrics=None,
    usage=None,
    error_message: str = "",
) -> VoiceSessionStatus:
    return VoiceSessionStatus(
        active=active,
        session_id=uuid4() if active else None,
        owner_person_id="person" if active else "",
        state=state,
        elapsed_seconds=elapsed,
        model_display_name="Qwen3.5 Flash",
        metrics=metrics or {},
        usage=usage or {},
        error_message=error_message,
    )


async def eventually(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached")


def test_voice_status_renderer_keeps_terminal_statistics_compact() -> None:
    rendered = OopzVoiceStatusRenderer().render(
        status(
            VoiceSessionState.CLOSED,
            active=False,
            elapsed=512,
            metrics={
                "voice_responses_drained": 17,
                "voice_task_control_calls": 4,
                "voice_task_notifications_presented": 3,
                "voice_provider_reconnects": 1,
                "voice_media_reconnects": 1,
            },
            usage={"total_tokens": 12_400},
        )
    )

    assert rendered == (
        "🎵 **语音会话结束** · Qwen3.5 Flash\n"
        "08:32 · 17 轮 · 后台调用 4 · 通知 3 · 重连 2 · 12.4k tokens"
    )
    assert len(rendered) < 2000


def test_voice_status_renderer_mentions_active_music_mix() -> None:
    rendered = OopzVoiceStatusRenderer().render(
        status(
            VoiceSessionState.SPEAKING,
            metrics={"audio_music_participant_active": 1},
        )
    )

    assert "与音乐混流中" in rendered


@pytest.mark.asyncio
async def test_voice_status_message_coalesces_updates_into_one_editable_message() -> None:
    gateway = FakeEditableGateway()
    display = OopzVoiceStatusMessage(
        gateway,
        address(),
        edit_interval_seconds=0.001,
        heartbeat_seconds=0.05,
    )
    await display.open()
    display.emit(status(VoiceSessionState.LISTENING))
    await eventually(lambda: bool(gateway.replaced))

    display.emit(status(VoiceSessionState.USER_SPEAKING, elapsed=13))
    display.emit(status(VoiceSessionState.THINKING, elapsed=14))
    display.emit(status(VoiceSessionState.SPEAKING, elapsed=15))
    display.emit(
        status(
            VoiceSessionState.CLOSED,
            active=False,
            elapsed=16,
            metrics={"voice_responses_drained": 2},
            usage={"input_tokens": 4, "output_tokens": 3},
        )
    )
    await display.aclose()

    assert len(gateway.created) == 1
    assert "正在启动" in gateway.created[0]
    assert "正在听" in gateway.replaced[0]
    assert "语音会话结束" in gateway.replaced[-1]
    assert "2 轮" in gateway.replaced[-1]
    assert "7 tokens" in gateway.replaced[-1]
    assert len(gateway.replaced) < 5


@pytest.mark.asyncio
async def test_voice_status_message_sends_terminal_fallback_after_edit_failure() -> None:
    gateway = FakeEditableGateway()
    gateway.replace_error = ValueError("fixture deterministic edit rejection")
    display = OopzVoiceStatusMessage(
        gateway,
        address(),
        edit_interval_seconds=0.001,
        heartbeat_seconds=0.05,
    )
    await display.open()
    display.emit(
        status(
            VoiceSessionState.FAILED,
            active=False,
            error_message="上游连接失败，请稍后重试。",
        )
    )
    await display.aclose()

    assert len(gateway.created) == 2
    assert "上游连接失败" in gateway.created[-1]


@pytest.mark.asyncio
async def test_voice_status_close_continues_after_caller_cancellation() -> None:
    gateway = FakeEditableGateway()
    gateway.replace_started = asyncio.Event()
    gateway.replace_release = asyncio.Event()
    display = OopzVoiceStatusMessage(
        gateway,
        address(),
        edit_interval_seconds=0.001,
        heartbeat_seconds=0.05,
    )
    await display.open()
    display.emit(status(VoiceSessionState.CLOSED, active=False))

    closing = asyncio.create_task(display.aclose())
    await gateway.replace_started.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert display._worker is not None
    assert display._worker.cancelled() is False
    assert display._close_task is not None
    assert display._close_task.cancelled() is False
    gateway.replace_release.set()
    await asyncio.wait_for(display._close_task, timeout=0.1)

    assert display._worker is None
    assert len(gateway.replaced) == 1
    assert "语音会话结束" in gateway.replaced[0]
    await display.aclose()


@pytest.mark.asyncio
async def test_voice_status_fallback_continues_after_caller_cancellation() -> None:
    gateway = FakeEditableGateway()
    gateway.replace_error = ValueError("fixture deterministic edit rejection")
    gateway.fallback_started = asyncio.Event()
    gateway.fallback_release = asyncio.Event()
    display = OopzVoiceStatusMessage(
        gateway,
        address(),
        edit_interval_seconds=0.001,
        heartbeat_seconds=0.05,
    )
    await display.open()
    display.emit(status(VoiceSessionState.FAILED, active=False, error_message="连接失败"))

    closing = asyncio.create_task(display.aclose())
    await gateway.fallback_started.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert display._close_task is not None
    assert display._close_task.cancelled() is False
    gateway.fallback_release.set()
    await asyncio.wait_for(display._close_task, timeout=0.1)

    assert display._fallback_sent is True
    assert len(gateway.created) == 2
    assert "连接失败" in gateway.created[-1]


@pytest.mark.asyncio
async def test_voice_status_close_retries_failed_terminal_fallback() -> None:
    gateway = FakeEditableGateway()
    gateway.replace_error = ValueError("fixture deterministic edit rejection")
    gateway.fallback_error = RuntimeError("fixture fallback failure")
    display = OopzVoiceStatusMessage(
        gateway,
        address(),
        edit_interval_seconds=0.001,
        heartbeat_seconds=0.05,
    )
    await display.open()
    display.emit(status(VoiceSessionState.FAILED, active=False, error_message="连接失败"))

    await display.aclose()

    assert display._fallback_sent is False
    assert display._close_complete is False
    first_close_task = display._close_task
    gateway.fallback_error = None
    await display.aclose()

    assert display._close_task is not first_close_task
    assert display._fallback_sent is True
    assert display._close_complete is True
    assert len(gateway.created) == 2

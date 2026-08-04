from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from cywl_oopz.features.voice import runtime as runtime_module
from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceProviderDisconnectedError
from cywl_oopz.features.voice.events import (
    VoiceAssistantAudio,
    VoiceProviderFailed,
    VoiceResponseCancelled,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionReady,
    VoiceToolCall,
    VoiceTranscriptFinal,
    VoiceUserSpeechStarted,
    VoiceUserSpeechStopped,
)
from cywl_oopz.features.voice.models import (
    PcmChunk,
    RemoteAudioFrame,
    VoiceAudioFormat,
    VoiceChannelKey,
    VoiceMediaEndReason,
    VoiceProviderCapabilities,
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceStopReason,
    VoiceTaskNotification,
    VoiceTaskNotificationStatus,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.features.voice.runtime import RealtimeVoiceSessionRuntimeImpl
from cywl_oopz.features.voice.settings import VoiceTurnRole
from cywl_oopz.integrations.voice.fake import (
    FakeRealtimeVoiceProvider,
    FakeVoiceAccessGateway,
    FakeVoiceConfigurationRepository,
    FakeVoiceMediaGateway,
    FakeVoiceSessionRepository,
)
from cywl_oopz.settings import VoiceSettings


def settings(**overrides: str) -> VoiceSettings:
    return VoiceSettings.from_mapping(
        {
            "CYWL_VOICE_ENABLED": "true",
            "CYWL_VOICE_START_TIMEOUT_SECONDS": "1",
            "CYWL_VOICE_PROVIDER_CONNECT_ATTEMPTS": "2",
            "CYWL_VOICE_IDLE_TIMEOUT_SECONDS": "30",
            "CYWL_VOICE_MAX_SESSION_SECONDS": "60",
            **overrides,
        }
    )


class FakeTaskMailbox:
    def __init__(self) -> None:
        self.pending: list[VoiceTaskNotification] = []
        self.signal = asyncio.Event()
        self.claim_calls = 0
        self.presented: list[tuple[VoiceTaskNotification, ...]] = []
        self.deferred: list[tuple] = []
        self.marked_presented: list[tuple] = []
        self.delivery_succeeds = True
        self.mark_presented_failures = 0

    async def wait(self, owner_person_id: str, timeout_seconds: float) -> bool:
        del owner_person_id
        try:
            async with asyncio.timeout(timeout_seconds):
                await self.signal.wait()
        except TimeoutError:
            return False
        self.signal.clear()
        return True

    async def claim(self, session_id, limit: int):
        del session_id
        self.claim_calls += 1
        claimed = tuple(self.pending[:limit])
        del self.pending[:limit]
        return claimed

    async def present_text(self, notices):
        self.presented.append(notices)
        return self.delivery_succeeds

    async def mark_presented(self, task_ids):
        if self.mark_presented_failures:
            self.mark_presented_failures -= 1
            raise RuntimeError("fixture persistence failure")
        self.marked_presented.append(task_ids)

    async def defer(self, task_ids):
        self.deferred.append(task_ids)


async def runtime_fixture(
    task_controls=None,
    task_mailbox=None,
    capabilities=None,
    status_sink=None,
    settings_overrides: dict[str, str] | None = None,
):
    access = FakeVoiceAccessGateway()
    channel = VoiceChannelKey("area", "voice")
    lease = await access.try_acquire(channel, "conversation:test")
    assert lease is not None
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "person", channel
    )
    configuration = replace(
        configuration,
        channel=replace(configuration.channel, idle_timeout_seconds=30),
    )
    descriptor = VoiceSessionDescriptor(
        uuid4(),
        "person",
        channel,
        VoiceTextAddress("area", "text"),
    )
    context = VoiceSessionRuntimeContext(descriptor, lease, configuration, status_sink)
    media = FakeVoiceMediaGateway()
    sessions = FakeVoiceSessionRepository()
    providers: list[FakeRealtimeVoiceProvider] = []

    def build_provider(_context):
        provider = FakeRealtimeVoiceProvider(capabilities)
        providers.append(provider)
        return provider

    runtime = RealtimeVoiceSessionRuntimeImpl(
        context,
        settings(**(settings_overrides or {})),
        media,
        sessions,
        build_provider,
        task_controls,
        task_mailbox,
    )
    return runtime, media, sessions, providers


async def wait_until(predicate, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def notification(sequence: int) -> VoiceTaskNotification:
    return VoiceTaskNotification(
        uuid4(),
        f"T{sequence}",
        VoiceTaskNotificationStatus.SUCCEEDED,
        f"查询任务 {sequence}",
        f"结果 {sequence}",
        "",
        VoiceTextAddress("area", "text"),
    )


@pytest.mark.asyncio
async def test_realtime_runtime_streams_audio_and_persists_final_turns() -> None:
    runtime, media_gateway, sessions, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting

    media = media_gateway.sessions[0]
    source = VoiceAudioFormat(48_000, 2, "f32le")
    frame_pcm = np.zeros((1_024, 2), dtype="<f4").tobytes()
    await media.push_input(RemoteAudioFrame(frame_pcm, source, 0, 1.0))
    await media.push_input(RemoteAudioFrame(frame_pcm, source, 1, 1.02))
    await provider_session.emit(VoiceTranscriptFinal("user", "你好", "user-1"))
    await provider_session.emit(VoiceResponseStarted("response-1"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x00" * 960, PROVIDER_OUTPUT_FORMAT, 20, generation=0),
            "response-1",
            "assistant-1",
        )
    )
    await provider_session.emit(
        VoiceTranscriptFinal("assistant", "你好呀", "assistant-1", "response-1")
    )
    await provider_session.emit(
        VoiceResponseCompleted("response-1", {"input_tokens": 3, "output_tokens": 2})
    )

    await wait_until(lambda: bool(provider_session.sent_audio))
    await wait_until(lambda: bool(media.outputs))
    await wait_until(lambda: len(sessions.turns) == 2)

    assert provider_session.sent_audio[0].format.sample_rate == 16_000
    assert media.outputs[0].format.sample_rate == 24_000
    assert sessions.turns == [
        (runtime._context.descriptor.session_id, 1, VoiceTurnRole.USER, "你好"),
        (runtime._context.descriptor.session_id, 2, VoiceTurnRole.ASSISTANT, "你好呀"),
    ]
    assert runtime.state is VoiceSessionState.LISTENING
    assert runtime.stats.provider_connect_attempts == 1
    assert runtime.stats.provider_connections == 1
    assert runtime.stats.initial_provider_ready_ms > 0
    assert runtime.stats.first_final_transcript_ms > 0
    assert runtime.stats.first_provider_audio_ms > 0
    assert runtime.stats.first_oopz_output_ms > 0
    assert runtime.stats.max_input_queue_depth >= 1
    assert runtime.stats.max_output_queue_depth >= 1

    await runtime.request_stop(VoiceStopReason.COMMAND)
    result = await runtime.wait_finished()
    assert result.reason is VoiceStopReason.COMMAND
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert result.metrics["voice_first_oopz_output_ms"] > 0
    await runtime.aclose()
    assert media.closed is True
    assert providers[0].closed is True


@pytest.mark.asyncio
async def test_realtime_runtime_stop_budget_cancels_stalled_transports_and_pumps() -> None:
    runtime, media_gateway, _, providers = await runtime_fixture()
    runtime._settings = settings(CYWL_VOICE_STOP_TIMEOUT_SECONDS="0.05")
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    media = media_gateway.sessions[0]
    provider_close_cancelled = asyncio.Event()
    media_close_cancelled = asyncio.Event()
    never = asyncio.Event()
    owned_tasks = tuple(runtime._tasks)

    async def stalled_provider_close() -> None:
        try:
            await never.wait()
        finally:
            providers[0].closed = True
            provider_close_cancelled.set()

    async def stalled_media_close() -> None:
        try:
            await never.wait()
        finally:
            media.closed = True
            media_close_cancelled.set()

    providers[0].aclose = stalled_provider_close
    media.aclose = stalled_media_close
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()

    started_at = time.monotonic()
    async with asyncio.timeout(2):
        await runtime.aclose()
    elapsed = time.monotonic() - started_at
    await asyncio.sleep(0)

    assert elapsed < 2
    assert provider_close_cancelled.is_set()
    assert media_close_cancelled.is_set()
    assert all(task.done() for task in owned_tasks)
    assert not runtime._tasks
    assert runtime.state is VoiceSessionState.CLOSED


@pytest.mark.asyncio
async def test_realtime_runtime_reports_bounded_input_queue_pressure() -> None:
    runtime, media_gateway, _, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    media = media_gateway.sessions[0]
    source = VoiceAudioFormat(48_000, 2, "f32le")
    frame_pcm = np.zeros((1_024, 2), dtype="<f4").tobytes()

    for sequence in range(30):
        await media.push_input(
            RemoteAudioFrame(
                frame_pcm,
                source,
                sequence,
                1.0 + sequence * 0.02,
                source_dropped_frames=2 if sequence == 0 else 0,
            )
        )
    await wait_until(lambda: runtime.stats.input_packets_dropped > 0)

    assert runtime.stats.max_input_queue_depth == runtime._input.max_chunks
    assert runtime.stats.source_audio_frames_dropped == 2
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting
    await runtime.request_stop(VoiceStopReason.COMMAND)
    result = await runtime.wait_finished()
    assert result.metrics["voice_input_packets_dropped"] > 0
    assert result.metrics["voice_max_input_queue_depth"] == runtime._input.max_chunks
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_emits_nonblocking_state_snapshots() -> None:
    class RecordingRuntimeStatusSink:
        def __init__(self) -> None:
            self.statuses = []

        def emit(self, status) -> None:
            self.statuses.append(status)

    sink = RecordingRuntimeStatusSink()
    runtime, _, _, providers = await runtime_fixture(status_sink=sink)
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    await provider_session.emit(VoiceUserSpeechStarted())
    await provider_session.emit(VoiceUserSpeechStopped())
    await provider_session.emit(VoiceResponseStarted("response-status"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x00" * 960, PROVIDER_OUTPUT_FORMAT, 20, generation=0),
            "response-status",
            "item-status",
        )
    )
    await wait_until(lambda: sink.statuses[-1].state is VoiceSessionState.SPEAKING)

    states = [item.state for item in sink.statuses]
    assert states[:4] == [
        VoiceSessionState.LISTENING,
        VoiceSessionState.USER_SPEAKING,
        VoiceSessionState.THINKING,
        VoiceSessionState.THINKING,
    ]
    assert sink.statuses[-1].stats.responses_started == 1
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()
    assert sink.statuses[-1].state is VoiceSessionState.CLOSED


@pytest.mark.asyncio
async def test_task_notification_waits_for_user_turn_then_uses_text_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_SILENCE_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    runtime, _, _, providers = await runtime_fixture(task_mailbox=mailbox)
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    await provider_session.emit(VoiceUserSpeechStarted())
    await wait_until(lambda: runtime.state is VoiceSessionState.USER_SPEAKING)

    initial_claims = mailbox.claim_calls
    mailbox.pending.append(notification(1))
    mailbox.signal.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert mailbox.claim_calls == initial_claims

    await provider_session.emit(VoiceUserSpeechStopped())
    await provider_session.emit(VoiceResponseStarted("response-turn"))
    await provider_session.emit(VoiceResponseCompleted("response-turn"))
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)
    mailbox.signal.set()
    await wait_until(lambda: runtime.stats.task_notifications_presented == 1)

    assert [item.alias for item in mailbox.presented[0]] == ["T1"]
    assert runtime.stats.task_notifications_claimed == 1
    assert runtime.stats.task_notifications_presented == 1
    assert runtime.stats.task_notifications_text_fallback == 1
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_notifications_are_bounded_to_three_per_active_session_batch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    mailbox.pending.extend(notification(index) for index in range(1, 5))
    runtime, _, _, providers = await runtime_fixture(task_mailbox=mailbox)
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: bool(mailbox.presented))
    assert len(mailbox.presented[0]) == 3
    assert [item.alias for item in mailbox.presented[0]] == ["T1", "T2", "T3"]

    mailbox.signal.set()
    await wait_until(lambda: len(mailbox.presented) == 2)
    assert [item.alias for item in mailbox.presented[1]] == ["T4"]
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_notification_delivery_failure_is_deferred(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COOLDOWN_SECONDS", 30)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    mailbox.delivery_succeeds = False
    notice = notification(1)
    mailbox.pending.append(notice)
    runtime, _, _, providers = await runtime_fixture(task_mailbox=mailbox)
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: runtime.stats.task_notifications_deferred == 1)
    assert len(mailbox.presented) == 1
    assert runtime.stats.task_notifications_presented == 0
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_capability_strategy_marks_presented_only_after_proactive_response_starts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    notice = notification(1)
    mailbox.pending.append(notice)
    capabilities = VoiceProviderCapabilities(
        context_injection=True,
        proactive_response=True,
    )
    runtime, _, _, providers = await runtime_fixture(
        task_mailbox=mailbox,
        capabilities=capabilities,
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    provider_session = providers[0].sessions[0]
    await wait_until(lambda: bool(provider_session.proactive_items))
    assert "[CYWL_INTERNAL_TASK_EVENT v1]" in provider_session.proactive_items[0].text
    assert mailbox.marked_presented == []
    assert mailbox.presented == []
    await provider_session.emit(VoiceResponseStarted("proactive-response"))
    await wait_until(lambda: runtime.stats.task_notifications_presented == 1)
    assert mailbox.marked_presented == [(notice.task_id,)]
    assert runtime.stats.task_notifications_deferred == 0
    await provider_session.emit(VoiceResponseCompleted("proactive-response"))
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_failed_proactive_request_returns_notification_to_mailbox(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    notice = notification(1)
    mailbox.pending.append(notice)
    capabilities = VoiceProviderCapabilities(
        context_injection=True,
        proactive_response=True,
    )
    runtime, _, _, providers = await runtime_fixture(
        task_mailbox=mailbox,
        capabilities=capabilities,
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    provider_session.proactive_error = RuntimeError("fixture injection failure")
    await provider_session.emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: runtime.stats.task_notifications_deferred == 1)
    assert mailbox.deferred == [(notice.task_id,)]
    assert mailbox.marked_presented == []
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_proactive_notification_confirmation_retries_transient_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_PERSIST_RETRY_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    mailbox.mark_presented_failures = 2
    notice = notification(1)
    mailbox.pending.append(notice)
    capabilities = VoiceProviderCapabilities(
        context_injection=True,
        proactive_response=True,
    )
    runtime, _, _, providers = await runtime_fixture(
        task_mailbox=mailbox,
        capabilities=capabilities,
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: bool(provider_session.proactive_items))
    await provider_session.emit(VoiceResponseStarted("proactive-response"))
    await wait_until(lambda: runtime.stats.task_notifications_presented == 1)

    assert mailbox.mark_presented_failures == 0
    assert mailbox.marked_presented == [(notice.task_id,)]
    assert runtime.stats.task_notifications_deferred == 0
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_retryable_disconnect_defers_unstarted_proactive_notification(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    notice = notification(1)
    mailbox.pending.append(notice)
    capabilities = VoiceProviderCapabilities(
        context_injection=True,
        proactive_response=True,
    )
    runtime, _, _, providers = await runtime_fixture(
        task_mailbox=mailbox,
        capabilities=capabilities,
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: bool(first_session.proactive_items))
    await first_session.emit(VoiceProviderFailed("connection_closed", True))
    await wait_until(lambda: len(providers) == 2 and bool(providers[1].sessions))

    assert mailbox.deferred == [(notice.task_id,)]
    assert mailbox.marked_presented == []
    assert runtime.stats.task_notifications_deferred == 1
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_close_defers_unstarted_proactive_notification(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    notice = notification(1)
    mailbox.pending.append(notice)
    capabilities = VoiceProviderCapabilities(
        context_injection=True,
        proactive_response=True,
    )
    runtime, _, _, providers = await runtime_fixture(
        task_mailbox=mailbox,
        capabilities=capabilities,
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting

    await wait_until(lambda: bool(provider_session.proactive_items))
    await runtime.aclose()

    assert mailbox.deferred == [(notice.task_id,)]
    assert mailbox.marked_presented == []


@pytest.mark.asyncio
async def test_realtime_task_control_does_not_block_audio_or_duplicate_calls() -> None:
    class GatedTaskControls:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def execute(self, descriptor, call_id, name, arguments):
            del descriptor, call_id, name, arguments
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return {"ok": True, "accepted": True, "task": "T1"}

    controls = GatedTaskControls()
    runtime, media_gateway, _, providers = await runtime_fixture(controls)
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting

    call = VoiceToolCall("call-1", "delegate_agent_task", {"objective": "查演出"})
    await provider_session.emit(call)
    await provider_session.emit(call)
    await controls.started.wait()

    source = VoiceAudioFormat(48_000, 2, "f32le")
    pcm = np.zeros((1_024, 2), dtype="<f4").tobytes()
    await media_gateway.sessions[0].push_input(RemoteAudioFrame(pcm, source, 0, 1.0))
    await media_gateway.sessions[0].push_input(RemoteAudioFrame(pcm, source, 1, 1.02))
    await wait_until(lambda: bool(provider_session.sent_audio))
    assert provider_session.tool_outputs == []
    assert controls.calls == 1

    controls.release.set()
    await wait_until(lambda: bool(provider_session.tool_outputs))
    assert provider_session.tool_outputs == [
        ("call-1", {"ok": True, "accepted": True, "task": "T1"})
    ]
    assert runtime.stats.task_control_calls == 1
    assert runtime.stats.task_control_failures == 0
    await provider_session.emit(VoiceUserSpeechStarted())
    await provider_session.emit(VoiceUserSpeechStopped())
    await wait_until(lambda: runtime.state is VoiceSessionState.THINKING)

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_task_control_timeout_returns_error_without_ending_session() -> None:
    class StuckTaskControls:
        async def execute(self, descriptor, call_id, name, arguments):
            del descriptor, call_id, name, arguments
            await asyncio.Event().wait()

    runtime, _, _, providers = await runtime_fixture(StuckTaskControls())
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting

    await provider_session.emit(VoiceToolCall("call-timeout", "get_agent_task", {"task": "T1"}))
    async with asyncio.timeout(0.5):
        while not provider_session.tool_outputs:
            await asyncio.sleep(0.01)

    assert provider_session.tool_outputs == [
        ("call-timeout", {"ok": False, "code": "temporarily_unavailable"})
    ]
    assert runtime.state is VoiceSessionState.LISTENING
    assert runtime.stats.task_control_failures == 1
    assert 100 <= runtime.stats.last_task_control_ms < 300

    await runtime.request_stop(VoiceStopReason.COMMAND)
    result = await runtime.wait_finished()
    assert result.metrics["voice_task_control_failures"] == 1
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_barge_in_flushes_then_drops_late_audio() -> None:
    runtime, media_gateway, sessions, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    media = media_gateway.sessions[0]
    operation_order: list[str] = []
    original_flush = media.flush_output
    original_interrupt = provider_session.interrupt

    async def traced_flush():
        operation_order.append("flush")
        return await original_flush()

    async def traced_interrupt(cursor):
        operation_order.append("interrupt")
        await original_interrupt(cursor)

    media.flush_output = traced_flush
    provider_session.interrupt = traced_interrupt

    await provider_session.emit(VoiceResponseStarted("response-1"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x01\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
            "response-1",
            "item-1",
        )
    )
    await provider_session.emit(
        VoiceTranscriptFinal("assistant", "这段尾音不会进入恢复上下文", "item-1", "response-1")
    )
    await wait_until(lambda: len(media.outputs) == 1)

    await provider_session.emit(VoiceUserSpeechStarted())
    await wait_until(lambda: len(provider_session.interruptions) == 1)
    assert media.flushes
    assert operation_order == ["flush", "interrupt"]
    assert runtime.state is VoiceSessionState.USER_SPEAKING
    assert runtime.stats.barge_in_count == 1
    assert runtime.stats.max_barge_in_flush_ms < 200
    output_count = len(media.outputs)

    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x02\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
            "response-1",
            "item-1",
        )
    )
    await provider_session.emit(VoiceResponseCancelled("response-1"))
    await provider_session.emit(VoiceUserSpeechStarted())
    await wait_until(lambda: runtime.stats.duplicate_speech_started == 1)
    assert len(media.outputs) == output_count
    assert len(provider_session.interruptions) == 1
    assert sessions.turns == []
    assert runtime.stats.late_audio_dropped >= 1
    assert runtime.stats.interrupted_transcripts_dropped == 1

    await provider_session.emit(VoiceUserSpeechStopped())
    await provider_session.emit(VoiceResponseStarted("response-2"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x03\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
            "response-2",
            "item-2",
        )
    )
    await provider_session.emit(VoiceTranscriptFinal("assistant", "新回答", "item-2", "response-2"))
    await provider_session.emit(VoiceResponseCompleted("response-2"))
    await wait_until(lambda: len(media.outputs) == output_count + 1)
    await wait_until(lambda: len(sessions.turns) == 1)
    assert sessions.turns[0][3] == "新回答"

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_recovers_after_interrupt_send_failure() -> None:
    runtime, media_gateway, sessions, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting
    media = media_gateway.sessions[0]

    await first_session.emit(VoiceResponseStarted("response-failing-cancel"))
    await first_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x01\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
            "response-failing-cancel",
            "item-failing-cancel",
        )
    )
    await wait_until(lambda: bool(media.outputs))
    first_session.interrupt_error = VoiceProviderDisconnectedError("fixture disconnect")
    await first_session.emit(VoiceUserSpeechStarted())

    await wait_until(lambda: len(providers) == 2 and bool(providers[1].sessions))
    assert media.flushes
    assert runtime.state is VoiceSessionState.RECOVERING
    assert sessions.recovering == [runtime._context.descriptor.session_id]
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await wait_until(lambda: runtime.state is VoiceSessionState.USER_SPEAKING)

    output_count = len(media.outputs)
    await (
        providers[1]
        .sessions[0]
        .emit(
            VoiceAssistantAudio(
                PcmChunk(b"\x02\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
                "response-failing-cancel",
                "item-failing-cancel",
            )
        )
    )
    await wait_until(lambda: runtime.stats.late_audio_dropped >= 1)
    assert len(media.outputs) == output_count

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_interrupts_completed_but_not_drained_playout_locally() -> None:
    runtime, media_gateway, sessions, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    media = media_gateway.sessions[0]
    media.drain_gate = asyncio.Event()

    await provider_session.emit(VoiceResponseStarted("response-buffered"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x01\x00" * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
            "response-buffered",
            "item-buffered",
        )
    )
    await provider_session.emit(
        VoiceTranscriptFinal(
            "assistant",
            "尚未完整播放",
            "item-buffered",
            "response-buffered",
        )
    )
    await provider_session.emit(VoiceResponseCompleted("response-buffered"))
    await wait_until(lambda: media.drain_count == 1)

    await provider_session.emit(VoiceUserSpeechStarted())
    await wait_until(lambda: runtime.state is VoiceSessionState.USER_SPEAKING)

    assert media.flushes
    assert provider_session.interruptions == []
    assert sessions.turns == []
    assert runtime.stats.interrupted_transcripts_dropped == 1
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_survives_twenty_generation_barriers() -> None:
    runtime, media_gateway, _, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    media = media_gateway.sessions[0]

    for index in range(20):
        response_id = f"response-{index}"
        item_id = f"item-{index}"
        await provider_session.emit(VoiceResponseStarted(response_id))
        await provider_session.emit(
            VoiceAssistantAudio(
                PcmChunk(bytes([index, 0]) * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
                response_id,
                item_id,
            )
        )
        await wait_until(lambda: len(media.outputs) == index + 1)
        await provider_session.emit(VoiceUserSpeechStarted())
        await wait_until(lambda: len(provider_session.interruptions) == index + 1)
        await provider_session.emit(
            VoiceAssistantAudio(
                PcmChunk(bytes([index + 1, 0]) * 480, PROVIDER_OUTPUT_FORMAT, 20, 0),
                response_id,
                item_id,
            )
        )
        await provider_session.emit(VoiceUserSpeechStopped())

    await wait_until(lambda: runtime.stats.late_audio_dropped >= 20)
    assert len(media.outputs) == 20
    assert len(media.flushes) == 20
    assert runtime.stats.barge_in_count == 20
    assert runtime.stats.max_barge_in_flush_ms < 200

    await runtime.request_stop(VoiceStopReason.COMMAND)
    result = await runtime.wait_finished()
    assert result.metrics["voice_barge_in_count"] == 20
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_recovers_retryable_provider_disconnect() -> None:
    runtime, _, sessions, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await providers[0].sessions[0].emit(VoiceProviderFailed("connection_closed", True))
    await wait_until(lambda: len(providers) == 2 and bool(providers[1].sessions))
    assert runtime.state is VoiceSessionState.RECOVERING
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)

    assert sessions.recovering == [runtime._context.descriptor.session_id]
    assert sessions.active == [runtime._context.descriptor.session_id]
    assert providers[0].closed is True
    assert runtime.stats.provider_connections == 2
    assert runtime.stats.provider_reconnects == 1
    assert runtime.stats.provider_recovery_failures == 0
    assert runtime.stats.last_provider_recovery_ms > 0
    assert runtime.stats.max_provider_recovery_ms >= runtime.stats.last_provider_recovery_ms

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_recovery_metrics_ignore_persistence_warning() -> None:
    runtime, _, sessions, providers = await runtime_fixture()

    async def fail_mark_recovering(session_id):
        del session_id
        raise RuntimeError("fixture persistence failure")

    sessions.mark_recovering = fail_mark_recovering
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await providers[0].sessions[0].emit(VoiceProviderFailed("connection_closed", True))
    await wait_until(lambda: len(providers) == 2 and bool(providers[1].sessions))
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)

    assert runtime.stats.provider_reconnects == 1
    assert runtime.stats.provider_recovery_failures == 0
    assert runtime.stats.last_provider_recovery_ms > 0
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_counts_exhausted_provider_recovery() -> None:
    runtime, _, _, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting

    class FailingProvider(FakeRealtimeVoiceProvider):
        async def connect(self, descriptor):
            del descriptor
            raise VoiceProviderDisconnectedError("fixture reconnect failure")

    runtime._settings = settings(CYWL_VOICE_PROVIDER_CONNECT_ATTEMPTS="1")
    runtime._provider_builder = lambda context: FailingProvider()
    await first_session.emit(VoiceProviderFailed("connection_closed", True))
    result = await runtime.wait_finished()

    assert result.reason is VoiceStopReason.PROVIDER_FAILED
    assert runtime.stats.provider_reconnects == 0
    assert runtime.stats.provider_recovery_failures == 1
    assert runtime.stats.provider_connect_attempts == 2
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_bounds_the_whole_provider_recovery_attempt() -> None:
    runtime, _, _, providers = await runtime_fixture(
        settings_overrides={
            "CYWL_VOICE_START_TIMEOUT_SECONDS": "0.05",
            "CYWL_VOICE_PROVIDER_CONNECT_ATTEMPTS": "5",
        }
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting
    connect_cancelled = asyncio.Event()
    stalled_providers: list[FakeRealtimeVoiceProvider] = []

    class StalledProvider(FakeRealtimeVoiceProvider):
        async def connect(self, descriptor):
            del descriptor
            try:
                await asyncio.Event().wait()
            finally:
                connect_cancelled.set()

    def build_stalled_provider(context):
        del context
        provider = StalledProvider()
        stalled_providers.append(provider)
        return provider

    runtime._provider_builder = build_stalled_provider
    started_at = time.monotonic()
    await first_session.emit(VoiceProviderFailed("connection_closed", True))

    async with asyncio.timeout(0.5):
        result = await runtime.wait_finished()
    assert time.monotonic() - started_at < 0.5
    assert result.reason is VoiceStopReason.PROVIDER_FAILED
    assert connect_cancelled.is_set()
    assert all(provider.closed for provider in stalled_providers)
    assert runtime.stats.provider_reconnects == 0
    assert runtime.stats.provider_recovery_failures == 1
    assert runtime.stats.provider_connect_attempts == 2
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_stop_preempts_provider_recovery() -> None:
    runtime, _, _, providers = await runtime_fixture(
        settings_overrides={
            "CYWL_VOICE_START_TIMEOUT_SECONDS": "10",
            "CYWL_VOICE_PROVIDER_CONNECT_ATTEMPTS": "5",
        }
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting
    connect_started = asyncio.Event()
    connect_cancelled = asyncio.Event()

    class StalledProvider(FakeRealtimeVoiceProvider):
        async def connect(self, descriptor):
            del descriptor
            connect_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                connect_cancelled.set()

    runtime._provider_builder = lambda context: StalledProvider()
    await first_session.emit(VoiceProviderFailed("connection_closed", True))
    await connect_started.wait()
    await runtime.request_stop(VoiceStopReason.COMMAND)

    async with asyncio.timeout(0.5):
        result = await runtime.wait_finished()
    assert result.reason is VoiceStopReason.COMMAND
    assert connect_cancelled.is_set()
    assert runtime.state is VoiceSessionState.CLOSING
    assert runtime.stats.provider_recovery_failures == 0
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_reconnects_with_only_confirmed_bounded_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COALESCE_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_SILENCE_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "_NOTIFICATION_COOLDOWN_SECONDS", 0)
    mailbox = FakeTaskMailbox()
    runtime, _, _, providers = await runtime_fixture(task_mailbox=mailbox)
    built_contexts = []
    original_builder = runtime._provider_builder

    def capture_context(context):
        built_contexts.append(context)
        return original_builder(context)

    runtime._provider_builder = capture_context
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_session = providers[0].sessions[0]
    await first_session.emit(VoiceSessionReady())
    await starting

    await first_session.emit(VoiceTranscriptFinal("user", "帮我查演唱会", "user-1"))
    await first_session.emit(VoiceResponseStarted("response-1"))
    await first_session.emit(
        VoiceTranscriptFinal("assistant", "已经交给后台处理啦。", "item-1", "response-1")
    )
    await first_session.emit(VoiceResponseCompleted("response-1"))
    await wait_until(lambda: runtime.stats.responses_drained == 1)

    mailbox.pending.append(notification(1))
    mailbox.signal.set()
    await wait_until(lambda: bool(mailbox.presented))
    await wait_until(lambda: runtime.stats.task_notifications_presented == 1)

    await first_session.emit(VoiceResponseStarted("response-unplayed"))
    await first_session.emit(
        VoiceTranscriptFinal(
            "assistant",
            "这段没有完整播放，不能进入恢复上下文。",
            "item-unplayed",
            "response-unplayed",
        )
    )
    await first_session.emit(VoiceProviderFailed("connection_closed", True))
    await wait_until(lambda: len(providers) == 2 and len(built_contexts) == 2)

    recovery = built_contexts[1].recovery_context
    assert [(turn.role, turn.text) for turn in recovery.turns] == [
        ("user", "帮我查演唱会"),
        ("assistant", "已经交给后台处理啦。"),
    ]
    assert [(task.alias, task.status.value, task.summary) for task in recovery.tasks] == [
        ("T1", "succeeded", "结果 1")
    ]
    assert "没有完整播放" not in repr(recovery)

    await providers[1].sessions[0].emit(VoiceSessionReady())
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_recovers_owner_media_within_grace() -> None:
    runtime, media, _, providers = await runtime_fixture(
        settings_overrides={"CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": "1"}
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    provider_session = providers[0].sessions[0]
    await provider_session.emit(VoiceSessionReady())
    await starting
    first_media = media.sessions[0]

    await first_media.end_input(VoiceMediaEndReason.OWNER_UNPUBLISHED)
    await wait_until(lambda: len(media.sessions) == 2)
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)
    recovered_media = media.sessions[1]
    source = VoiceAudioFormat(48_000, 2, "f32le")
    frame_pcm = np.zeros((1_024, 2), dtype="<f4").tobytes()
    await recovered_media.push_input(RemoteAudioFrame(frame_pcm, source, 1, 1.0))
    await recovered_media.push_input(RemoteAudioFrame(frame_pcm, source, 2, 1.02))
    await wait_until(lambda: bool(provider_session.sent_audio))
    await provider_session.emit(VoiceResponseStarted("response-after-media-recovery"))
    await provider_session.emit(
        VoiceAssistantAudio(
            PcmChunk(b"\x00" * 960, PROVIDER_OUTPUT_FORMAT, 20, generation=0),
            "response-after-media-recovery",
            "item-after-media-recovery",
        )
    )
    await wait_until(lambda: bool(recovered_media.outputs))

    assert first_media.closed is True
    assert not first_media.outputs
    assert len(providers) == 1
    assert runtime.stats.media_reconnects == 1
    assert runtime.stats.media_recovery_failures == 0
    assert runtime.stats.last_media_recovery_ms > 0
    await runtime.request_stop(VoiceStopReason.COMMAND)
    assert (await runtime.wait_finished()).reason is VoiceStopReason.COMMAND
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_owner_leave_is_immediate_when_grace_is_disabled() -> None:
    runtime, media, _, providers = await runtime_fixture(
        settings_overrides={"CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": "0"}
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await media.sessions[0].end_input(VoiceMediaEndReason.OWNER_LEFT)

    assert (await runtime.wait_finished()).reason is VoiceStopReason.OWNER_LEFT
    assert len(media.sessions) == 1
    assert runtime.stats.media_reconnects == 0
    assert runtime.stats.media_recovery_failures == 0
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_ends_when_owner_media_grace_expires() -> None:
    runtime, media, _, providers = await runtime_fixture(
        settings_overrides={"CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": "1"}
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting
    reopen_cancelled = asyncio.Event()
    never = asyncio.Event()

    async def stalled_open(descriptor, lease):
        del descriptor, lease
        try:
            await never.wait()
        finally:
            reopen_cancelled.set()

    media.open = stalled_open
    await media.sessions[0].end_input(VoiceMediaEndReason.OWNER_LEFT)

    async with asyncio.timeout(2):
        result = await runtime.wait_finished()
    assert result.reason is VoiceStopReason.OWNER_LEFT
    assert reopen_cancelled.is_set()
    assert runtime.stats.media_reconnects == 0
    assert runtime.stats.media_recovery_failures == 1
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_stop_preempts_owner_media_recovery() -> None:
    runtime, media, _, providers = await runtime_fixture(
        settings_overrides={"CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": "10"}
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting
    reopen_started = asyncio.Event()
    reopen_cancelled = asyncio.Event()
    never = asyncio.Event()

    async def stalled_open(descriptor, lease):
        del descriptor, lease
        reopen_started.set()
        try:
            await never.wait()
        finally:
            reopen_cancelled.set()

    media.open = stalled_open
    await media.sessions[0].end_input(VoiceMediaEndReason.OWNER_LEFT)
    await reopen_started.wait()
    await wait_until(lambda: runtime.state is VoiceSessionState.RECOVERING)

    await runtime.request_stop(VoiceStopReason.COMMAND)
    assert (await runtime.wait_finished()).reason is VoiceStopReason.COMMAND
    async with asyncio.timeout(2):
        await runtime.aclose()

    assert reopen_cancelled.is_set()
    assert runtime._pending_media is None
    assert not runtime._tasks


@pytest.mark.asyncio
async def test_realtime_runtime_waits_for_media_and_provider_when_both_recover() -> None:
    runtime, media, sessions, providers = await runtime_fixture(
        settings_overrides={"CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": "2"}
    )
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    first_provider_session = providers[0].sessions[0]
    await first_provider_session.emit(VoiceSessionReady())
    await starting
    original_open = media.open
    reopen_started = asyncio.Event()
    allow_reopen = asyncio.Event()

    async def delayed_open(descriptor, lease):
        reopen_started.set()
        await allow_reopen.wait()
        return await original_open(descriptor, lease)

    media.open = delayed_open
    await media.sessions[0].end_input(VoiceMediaEndReason.OWNER_UNPUBLISHED)
    await reopen_started.wait()
    await first_provider_session.emit(VoiceProviderFailed("connection_closed", True))
    await wait_until(lambda: len(providers) == 2 and bool(providers[1].sessions))
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await wait_until(lambda: runtime._provider_ready.is_set())

    assert runtime.state is VoiceSessionState.RECOVERING
    assert not sessions.active
    allow_reopen.set()
    await wait_until(lambda: len(media.sessions) == 2)
    await wait_until(lambda: runtime.state is VoiceSessionState.LISTENING)
    assert runtime.stats.provider_reconnects == 1
    assert runtime.stats.media_reconnects == 1
    assert sessions.active == [runtime._context.descriptor.session_id]

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_retries_initial_provider_connection() -> None:
    access = FakeVoiceAccessGateway()
    channel = VoiceChannelKey("area", "voice")
    lease = await access.try_acquire(channel, "conversation:test")
    assert lease is not None
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "person", channel
    )
    context = VoiceSessionRuntimeContext(
        VoiceSessionDescriptor(
            uuid4(),
            "person",
            channel,
            VoiceTextAddress("area", "text"),
        ),
        lease,
        configuration,
    )

    class FailingProvider(FakeRealtimeVoiceProvider):
        async def connect(self, descriptor):
            del descriptor
            raise VoiceProviderDisconnectedError("fixture disconnect")

    providers = [FailingProvider(), FakeRealtimeVoiceProvider()]
    attempts = 0

    def build_provider(_context):
        nonlocal attempts
        provider = providers[attempts]
        attempts += 1
        return provider

    runtime = RealtimeVoiceSessionRuntimeImpl(
        context,
        settings(),
        FakeVoiceMediaGateway(),
        FakeVoiceSessionRepository(),
        build_provider,
    )
    starting = asyncio.create_task(runtime.start())
    await asyncio.sleep(0.3)
    assert providers[1].sessions
    await providers[1].sessions[0].emit(VoiceSessionReady())
    await starting

    assert attempts == 2
    assert providers[0].closed is True
    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_fails_start_immediately_on_fatal_provider_event() -> None:
    runtime, _, _, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))

    await providers[0].sessions[0].emit(VoiceProviderFailed("invalid_session", False))

    with pytest.raises(VoiceProviderDisconnectedError):
        await starting
    assert runtime.state is VoiceSessionState.FAILED
    await runtime.aclose()

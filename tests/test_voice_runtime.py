from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceProviderDisconnectedError
from cywl_oopz.features.voice.events import (
    VoiceAssistantAudio,
    VoiceProviderFailed,
    VoiceResponseCancelled,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionReady,
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
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceStopReason,
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


async def runtime_fixture():
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
    context = VoiceSessionRuntimeContext(descriptor, lease, configuration)
    media = FakeVoiceMediaGateway()
    sessions = FakeVoiceSessionRepository()
    providers: list[FakeRealtimeVoiceProvider] = []

    def build_provider(_context):
        provider = FakeRealtimeVoiceProvider()
        providers.append(provider)
        return provider

    runtime = RealtimeVoiceSessionRuntimeImpl(
        context,
        settings(),
        media,
        sessions,
        build_provider,
    )
    return runtime, media, sessions, providers


async def wait_until(predicate, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


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

    await runtime.request_stop(VoiceStopReason.COMMAND)
    result = await runtime.wait_finished()
    assert result.reason is VoiceStopReason.COMMAND
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    await runtime.aclose()
    assert media.closed is True
    assert providers[0].closed is True


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

    await runtime.request_stop(VoiceStopReason.COMMAND)
    await runtime.wait_finished()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_realtime_runtime_ends_when_owner_leaves() -> None:
    runtime, media, _, providers = await runtime_fixture()
    starting = asyncio.create_task(runtime.start())
    await wait_until(lambda: bool(providers and providers[0].sessions))
    await providers[0].sessions[0].emit(VoiceSessionReady())
    await starting

    await media.sessions[0].end_input(VoiceMediaEndReason.OWNER_LEFT)

    assert (await runtime.wait_finished()).reason is VoiceStopReason.OWNER_LEFT
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

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest
from oopz_sdk import PcmFormat, VoiceAudioEndReason, VoicePlaybackCursor
from oopz_sdk.models import AgoraAudioTrackData

from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceMediaTransportError
from cywl_oopz.features.voice.models import (
    PcmChunk,
    VoiceChannelKey,
    VoiceMediaEndReason,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.oopz.voice_media import OopzVoiceMediaGateway
from cywl_oopz.settings import VoiceSettings


class FakeLease:
    released = False

    async def release(self) -> bool:
        self.released = True
        return True


class FakeSubscription:
    def __init__(self, frames=()) -> None:
        self.frames = list(frames)
        self.end_reason = VoiceAudioEndReason.REMOTE_LEFT
        self.terminal_error = None
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        item = self.frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def wait_closed(self):
        return self.end_reason

    async def aclose(self) -> None:
        self.close_count += 1


class FakeOutput:
    def __init__(self) -> None:
        self.format = PcmFormat.s16le(sample_rate=48_000, channels=2)
        self.stats = SimpleNamespace(
            generation=0,
            accepted_samples=0,
            rendered_samples=0,
            buffered_samples=0,
        )
        self.writes: list[bytes] = []
        self.close_count = 0

    async def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)
        frames = len(pcm) // self.format.frame_width
        self.stats.accepted_samples += frames
        self.stats.buffered_samples += frames

    async def flush(self) -> VoicePlaybackCursor:
        cursor = VoicePlaybackCursor(
            self.stats.generation,
            self.stats.accepted_samples,
            self.stats.rendered_samples,
            self.stats.buffered_samples,
            self.format.sample_rate,
            1.0,
        )
        self.stats.generation += 1
        self.stats.accepted_samples = 0
        self.stats.rendered_samples = 0
        self.stats.buffered_samples = 0
        return cursor

    async def drain(self) -> VoicePlaybackCursor:
        self.stats.rendered_samples = self.stats.accepted_samples
        self.stats.buffered_samples = 0
        return VoicePlaybackCursor(
            self.stats.generation,
            self.stats.accepted_samples,
            self.stats.rendered_samples,
            0,
            self.format.sample_rate,
            2.0,
        )

    async def aclose(self) -> None:
        self.close_count += 1


class FakeVoice:
    def __init__(self, subscription: FakeSubscription, output: FakeOutput) -> None:
        self.subscription = subscription
        self.output = output
        self.subscribe_calls: list[tuple[str, dict[str, object]]] = []
        self.output_calls: list[tuple[PcmFormat, dict[str, int]]] = []
        self.output_error: Exception | None = None

    async def subscribe_person_audio(self, person_id: str, **options):
        self.subscribe_calls.append((person_id, options))
        return self.subscription

    async def open_pcm_output(self, format: PcmFormat, **options):
        self.output_calls.append((format, options))
        if self.output_error is not None:
            raise self.output_error
        return self.output


def settings() -> VoiceSettings:
    return VoiceSettings.from_mapping(
        {
            "CYWL_VOICE_ENABLED": "true",
            "CYWL_VOICE_START_TIMEOUT_SECONDS": "12",
            "CYWL_VOICE_OUTPUT_QUEUE_MS": "360",
            "CYWL_VOICE_OUTPUT_PREBUFFER_MS": "90",
        }
    )


def descriptor() -> VoiceSessionDescriptor:
    return VoiceSessionDescriptor(
        UUID("10000000-0000-0000-0000-000000000001"),
        "owner-person",
        VoiceChannelKey("area", "voice"),
        VoiceTextAddress("area", "text"),
    )


def sdk_frame() -> AgoraAudioTrackData:
    return AgoraAudioTrackData(
        uid=873288360,
        sequence=7,
        sample_rate=48_000,
        channels=2,
        samples_per_channel=4,
        data=b"\x00" * 32,
        browser_dropped_before=2,
        received_at_monotonic=123.5,
    )


def output_chunk() -> PcmChunk:
    return PcmChunk(b"\x00" * 960, PROVIDER_OUTPUT_FORMAT, 20, 0)


def long_output_chunk() -> PcmChunk:
    return PcmChunk(b"\x00" * 4_800, PROVIDER_OUTPUT_FORMAT, 100, 0)


@pytest.mark.asyncio
async def test_oopz_media_opens_only_owner_input_and_fixed_output_contract() -> None:
    subscription = FakeSubscription((sdk_frame(),))
    output = FakeOutput()
    voice = FakeVoice(subscription, output)
    gateway = OopzVoiceMediaGateway(SimpleNamespace(voice=voice), settings())

    media = await gateway.open(descriptor(), FakeLease())
    frames = [item async for item in media.input_frames()]

    assert voice.subscribe_calls == [
        (
            "owner-person",
            {
                "frame_size": 1024,
                "max_queue_size": 8,
                "wait_timeout": 12.0,
                "force_profile": True,
            },
        )
    ]
    format, options = voice.output_calls[0]
    assert format == PcmFormat.s16le(sample_rate=48_000, channels=2)
    assert options == {"prebuffer_ms": 40, "max_buffer_ms": 160}
    assert frames[0].sequence == 7
    assert frames[0].format.sample_rate == 48_000
    assert frames[0].format.channels == 2
    assert frames[0].captured_at_monotonic == 123.5
    assert frames[0].source_dropped_frames == 2

    written = await media.write_output(output_chunk())
    assert written.generation == 0
    assert written.accepted_samples == 480
    assert (await media.current_cursor()) == written
    drained = await media.drain_output()
    assert output.writes
    assert all(len(pcm) == 3_840 for pcm in output.writes)
    assert drained.accepted_samples == drained.rendered_samples == 480
    flushed = await media.flush_output()
    assert flushed == drained
    assert (await media.current_cursor()).generation == 1

    await media.aclose()
    await media.aclose()
    assert subscription.close_count == 1
    assert output.close_count == 1


@pytest.mark.asyncio
async def test_oopz_media_close_retries_both_resources_after_cancellation() -> None:
    subscription = FakeSubscription()
    output = FakeOutput()
    media = await OopzVoiceMediaGateway(
        SimpleNamespace(voice=FakeVoice(subscription, output)), settings()
    ).open(descriptor(), FakeLease())
    subscription_started = asyncio.Event()
    output_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def stalled_subscription_close() -> None:
        subscription.close_count += 1
        subscription_started.set()
        await allow_close.wait()

    async def stalled_output_close() -> None:
        output.close_count += 1
        output_started.set()
        await allow_close.wait()

    subscription.aclose = stalled_subscription_close
    output.aclose = stalled_output_close
    closing = asyncio.create_task(media.aclose())
    await subscription_started.wait()
    await output_started.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert media._closing is True
    assert media._closed is False
    assert media._subscription_closed is False
    assert media._output_closed is False
    allow_close.set()
    await media.aclose()

    assert subscription.close_count == 2
    assert output.close_count == 2
    assert media._closed is True


@pytest.mark.asyncio
async def test_oopz_media_close_retries_only_failed_resource() -> None:
    subscription = FakeSubscription()
    output = FakeOutput()
    media = await OopzVoiceMediaGateway(
        SimpleNamespace(voice=FakeVoice(subscription, output)), settings()
    ).open(descriptor(), FakeLease())
    remaining_failures = 1

    async def flaky_subscription_close() -> None:
        nonlocal remaining_failures
        subscription.close_count += 1
        if remaining_failures:
            remaining_failures -= 1
            raise RuntimeError("fixture close failure")

    subscription.aclose = flaky_subscription_close
    await media.aclose()

    assert media._closed is False
    assert media._subscription_closed is False
    assert media._output_closed is True
    assert output.close_count == 1
    await media.aclose()

    assert media._closed is True
    assert subscription.close_count == 2
    assert output.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_reason", "domain_reason"),
    tuple(
        zip(
            VoiceAudioEndReason,
            (
                VoiceMediaEndReason.CLOSED_BY_CALLER,
                VoiceMediaEndReason.OWNER_UNPUBLISHED,
                VoiceMediaEndReason.OWNER_LEFT,
                VoiceMediaEndReason.VOICE_LEFT,
                VoiceMediaEndReason.BACKEND_CLOSED,
                VoiceMediaEndReason.TRANSPORT_LOST,
                VoiceMediaEndReason.QUEUE_OVERFLOW,
            ),
            strict=True,
        )
    ),
)
async def test_oopz_media_maps_every_sdk_input_terminal_reason(sdk_reason, domain_reason) -> None:
    subscription = FakeSubscription()
    subscription.end_reason = sdk_reason
    if sdk_reason is VoiceAudioEndReason.TRANSPORT_LOST:
        subscription.terminal_error = RuntimeError("transport details")
    output = FakeOutput()
    media = await OopzVoiceMediaGateway(
        SimpleNamespace(voice=FakeVoice(subscription, output)), settings()
    ).open(descriptor(), FakeLease())

    terminal = await media.wait_input_closed()

    assert terminal.reason is domain_reason
    assert terminal.error_kind == (
        "RuntimeError" if sdk_reason is VoiceAudioEndReason.TRANSPORT_LOST else None
    )
    await media.aclose()


@pytest.mark.asyncio
async def test_oopz_media_closes_input_when_output_open_fails() -> None:
    subscription = FakeSubscription()
    voice = FakeVoice(subscription, FakeOutput())
    voice.output_error = RuntimeError("browser failed")
    gateway = OopzVoiceMediaGateway(SimpleNamespace(voice=voice), settings())

    with pytest.raises(VoiceMediaTransportError) as failure:
        await gateway.open(descriptor(), FakeLease())

    assert failure.value.operation == "open_output"
    assert failure.value.error_kind == "RuntimeError"
    assert subscription.close_count == 1


@pytest.mark.asyncio
async def test_oopz_media_maps_input_and_output_failures_without_sdk_leakage() -> None:
    subscription = FakeSubscription((RuntimeError("remote transport details"),))
    output = FakeOutput()
    media = await OopzVoiceMediaGateway(
        SimpleNamespace(voice=FakeVoice(subscription, output)), settings()
    ).open(descriptor(), FakeLease())

    with pytest.raises(VoiceMediaTransportError) as input_failure:
        _ = [item async for item in media.input_frames()]
    assert input_failure.value.operation == "input_frames"
    assert input_failure.value.error_kind == "RuntimeError"

    async def failed_write(_pcm: bytes) -> None:
        raise RuntimeError("output transport details")

    output.write = failed_write
    with pytest.raises(VoiceMediaTransportError) as output_failure:
        await media.write_output(long_output_chunk())
    assert output_failure.value.operation == "write_output"
    assert output_failure.value.error_kind == "RuntimeError"
    await media.aclose()


@pytest.mark.asyncio
async def test_oopz_media_rejects_released_lease_before_sdk_calls() -> None:
    lease = FakeLease()
    lease.released = True
    voice = FakeVoice(FakeSubscription(), FakeOutput())
    gateway = OopzVoiceMediaGateway(SimpleNamespace(voice=voice), settings())

    with pytest.raises(ValueError, match="active voice lease"):
        await gateway.open(descriptor(), lease)

    assert voice.subscribe_calls == []

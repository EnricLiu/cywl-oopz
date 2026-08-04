from __future__ import annotations

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
        self.format = PcmFormat.s16le()
        self.stats = SimpleNamespace(
            generation=3,
            accepted_samples=480,
            rendered_samples=240,
            buffered_samples=240,
        )
        self.writes: list[bytes] = []
        self.close_count = 0
        self.flush_cursor = VoicePlaybackCursor(4, 480, 360, 120, 24_000, 1.0)
        self.drain_cursor = VoicePlaybackCursor(4, 480, 480, 0, 24_000, 2.0)

    async def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    async def flush(self) -> VoicePlaybackCursor:
        return self.flush_cursor

    async def drain(self) -> VoicePlaybackCursor:
        return self.drain_cursor

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
    return PcmChunk(b"\x00" * 960, PROVIDER_OUTPUT_FORMAT, 20, 3)


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
    assert format == PcmFormat.s16le(sample_rate=24_000, channels=1)
    assert options == {"prebuffer_ms": 90, "max_buffer_ms": 360}
    assert frames[0].sequence == 7
    assert frames[0].format.sample_rate == 48_000
    assert frames[0].format.channels == 2
    assert frames[0].captured_at_monotonic == 123.5
    assert frames[0].source_dropped_frames == 2

    written = await media.write_output(output_chunk())
    assert output.writes == [b"\x00" * 960]
    assert written.generation == 3
    assert written.rendered_samples == 240
    assert (await media.current_cursor()) == written
    assert (await media.flush_output()).rendered_samples == 360
    assert (await media.drain_output()).buffered_samples == 0

    await media.aclose()
    await media.aclose()
    assert subscription.close_count == 1
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
        await media.write_output(output_chunk())
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

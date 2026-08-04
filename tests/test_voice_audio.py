from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cywl_oopz.features.voice.audio import (
    INPUT_PACKET_BYTES,
    PROVIDER_INPUT_FORMAT,
    PROVIDER_OUTPUT_FORMAT,
    VoiceAudioIngress,
    VoiceInputQueue,
    VoiceOutputTransitQueue,
)
from cywl_oopz.features.voice.errors import (
    VoiceAudioQueueClosedError,
    VoiceOutputBackpressureError,
)
from cywl_oopz.features.voice.models import PcmChunk, RemoteAudioFrame, VoiceAudioFormat


def frame(
    samples: np.ndarray,
    *,
    sequence: int,
    sample_rate: int = 48_000,
    channels: int = 1,
    source_drops: int = 0,
) -> RemoteAudioFrame:
    shaped = np.asarray(samples, dtype="<f4").reshape(-1, channels)
    return RemoteAudioFrame(
        shaped.tobytes(),
        VoiceAudioFormat(sample_rate, channels, "f32le"),
        sequence,
        float(sequence),
        source_drops,
    )


def convert(ingress: VoiceAudioIngress, frames: list[RemoteAudioFrame]) -> bytes:
    packets = [packet for item in frames for packet in ingress.push(item)]
    packets.extend(ingress.flush())
    assert all(packet.format == PROVIDER_INPUT_FORMAT for packet in packets)
    assert all(len(packet.pcm) == INPUT_PACKET_BYTES for packet in packets)
    return b"".join(packet.pcm for packet in packets)


def input_chunk(value: int = 0) -> PcmChunk:
    return PcmChunk(
        int(value).to_bytes(2, "little", signed=True) * 320,
        PROVIDER_INPUT_FORMAT,
        20,
        0,
    )


def output_chunk(generation: int, value: int = 0, duration_ms: int = 20) -> PcmChunk:
    samples = PROVIDER_OUTPUT_FORMAT.sample_rate * duration_ms // 1000
    return PcmChunk(
        int(value).to_bytes(2, "little", signed=True) * samples,
        PROVIDER_OUTPUT_FORMAT,
        duration_ms,
        generation,
    )


def test_streaming_resampler_is_continuous_across_awkward_sdk_frame_boundaries() -> None:
    sample_rate = 48_000
    positions = np.arange(sample_rate, dtype=np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 997 * positions / sample_rate)).astype(np.float32)
    single = convert(VoiceAudioIngress(), [frame(tone, sequence=0)])

    split_frames = []
    offset = 0
    sequence = 0
    for size in (997, 1024, 333, 2048, 777):
        while offset < tone.size:
            chunk = tone[offset : offset + size]
            split_frames.append(frame(chunk, sequence=sequence))
            offset += chunk.size
            sequence += 1
            if sequence % 5 == 0:
                break
    if offset < tone.size:
        split_frames.append(frame(tone[offset:], sequence=sequence))
    split_ingress = VoiceAudioIngress()
    split = convert(split_ingress, split_frames)

    assert split == single
    assert len(split) == 16_000 * 2
    assert split_ingress.stats.packets_emitted == 50
    assert split_ingress.stats.trailing_samples_dropped == 0


def test_ingress_downmixes_stereo_and_accounts_for_clipping_invalid_and_sequence_gaps() -> None:
    ingress = VoiceAudioIngress()
    left = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    stereo = np.column_stack((left, -left))
    packets = ingress.push(frame(stereo, sequence=3, sample_rate=16_000, channels=2))
    assert len(packets) == 1
    assert set(packets[0].pcm) == {0}

    unusual = np.zeros(320, dtype=np.float32)
    unusual[:4] = [2.0, -2.0, np.nan, np.inf]
    packets = ingress.push(frame(unusual, sequence=5, sample_rate=16_000, source_drops=2))
    assert len(packets) == 1
    assert ingress.push(frame(unusual, sequence=5, sample_rate=16_000)) == ()
    ingress.flush()

    assert ingress.stats.sequence_gap_frames == 1
    assert ingress.stats.source_reported_drops == 2
    assert ingress.stats.frames_discarded_out_of_order == 1
    assert ingress.stats.invalid_samples == 2
    assert ingress.stats.clipped_samples == 2


def test_ingress_flushes_resampler_and_drops_short_tail_on_format_change() -> None:
    ingress = VoiceAudioIngress()
    old = np.zeros(4_000, dtype=np.float32)
    new = np.zeros(320, dtype=np.float32)

    old_packets = ingress.push(frame(old, sequence=0, sample_rate=48_000))
    new_packets = ingress.push(frame(new, sequence=1, sample_rate=16_000))
    final_packets = ingress.flush()

    assert len(old_packets) == 3
    assert len(new_packets) == 2
    assert len(final_packets) == 0
    assert ingress.stats.format_changes == 1
    assert 0 < ingress.stats.trailing_samples_dropped < 320


@pytest.mark.asyncio
async def test_input_queue_drops_oldest_to_keep_realtime_audio_current() -> None:
    queue = VoiceInputQueue(40)
    assert queue.put(input_chunk(1)) is False
    assert queue.put(input_chunk(2)) is False
    assert queue.put(input_chunk(3)) is True

    assert int.from_bytes((await queue.get()).pcm[:2], "little", signed=True) == 2
    assert int.from_bytes((await queue.get()).pcm[:2], "little", signed=True) == 3
    assert queue.stats.dropped_oldest == 1
    assert queue.stats.delivered == 2
    await queue.aclose()
    with pytest.raises(VoiceAudioQueueClosedError):
        await queue.get()


@pytest.mark.asyncio
async def test_input_queue_close_wakes_a_waiting_consumer() -> None:
    queue = VoiceInputQueue(20)
    waiting = asyncio.create_task(queue.get())
    await asyncio.sleep(0)

    await queue.aclose()

    with pytest.raises(VoiceAudioQueueClosedError):
        await waiting


@pytest.mark.asyncio
async def test_output_flush_invalidates_blocked_old_generation_without_late_enqueue() -> None:
    queue = VoiceOutputTransitQueue(40)
    generation = await queue.start_generation()
    assert await queue.put(output_chunk(generation, 1)) is True
    assert await queue.put(output_chunk(generation, 2)) is True
    blocked = asyncio.create_task(queue.put(output_chunk(generation, 3)))
    await asyncio.sleep(0)
    assert blocked.done() is False

    next_generation = await queue.flush()

    assert next_generation == generation + 1
    assert await blocked is False
    assert queue.qsize == 0
    assert queue.buffered_duration_ms == 0
    assert queue.stats.flushed_chunks == 2
    assert queue.stats.stale_generation_dropped == 1
    assert await queue.put(output_chunk(next_generation, 4)) is True
    assert (await queue.get()).generation == next_generation
    await queue.aclose()


@pytest.mark.asyncio
async def test_output_queue_preserves_audio_and_fails_on_hard_backpressure_timeout() -> None:
    queue = VoiceOutputTransitQueue(20, put_timeout_seconds=0.01)
    generation = await queue.start_generation()
    await queue.put(output_chunk(generation, 1))

    with pytest.raises(VoiceOutputBackpressureError):
        await queue.put(output_chunk(generation, 2))

    assert queue.stats.backpressure_timeouts == 1
    assert int.from_bytes((await queue.get()).pcm[:2], "little", signed=True) == 1
    await queue.aclose()
    with pytest.raises(VoiceAudioQueueClosedError):
        await queue.get()


@pytest.mark.asyncio
async def test_output_queue_close_wakes_waiting_consumer_and_producer() -> None:
    queue = VoiceOutputTransitQueue(20)
    generation = await queue.start_generation()
    consumer = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    await queue.aclose()
    with pytest.raises(VoiceAudioQueueClosedError):
        await consumer

    with pytest.raises(VoiceAudioQueueClosedError):
        await queue.put(output_chunk(generation))
    with pytest.raises(VoiceAudioQueueClosedError):
        await queue.start_generation()

    producer_queue = VoiceOutputTransitQueue(20)
    generation = await producer_queue.start_generation()
    await producer_queue.put(output_chunk(generation, 1))
    producer = asyncio.create_task(producer_queue.put(output_chunk(generation, 2)))
    await asyncio.sleep(0)
    await producer_queue.aclose()
    with pytest.raises(VoiceAudioQueueClosedError):
        await producer

"""Stateful PCM conversion and bounded realtime transit queues."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import soxr

from .errors import VoiceAudioQueueClosedError, VoiceOutputBackpressureError
from .models import PcmChunk, RemoteAudioFrame, VoiceAudioFormat

PROVIDER_INPUT_FORMAT = VoiceAudioFormat(16_000, 1, "s16le")
PROVIDER_OUTPUT_FORMAT = VoiceAudioFormat(24_000, 1, "s16le")
INPUT_PACKET_DURATION_MS = 20
INPUT_PACKET_SAMPLES = 320
INPUT_PACKET_BYTES = 640


@dataclass(frozen=True, slots=True)
class VoiceIngressStats:
    """Cumulative synchronous ingress conversion counters."""

    frames_received: int = 0
    frames_discarded_out_of_order: int = 0
    source_reported_drops: int = 0
    sequence_gap_frames: int = 0
    format_changes: int = 0
    invalid_samples: int = 0
    clipped_samples: int = 0
    packets_emitted: int = 0
    trailing_samples_dropped: int = 0


class VoiceAudioIngress:
    """Downmix and continuously resample remote float PCM into exact 20 ms packets."""

    def __init__(self, *, resample_quality: str = "HQ") -> None:
        self._quality = resample_quality
        self._source_format: VoiceAudioFormat | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._pending_pcm = bytearray()
        self._previous_sequence: int | None = None
        self._stats = VoiceIngressStats()

    @property
    def stats(self) -> VoiceIngressStats:
        return self._stats

    def push(self, frame: RemoteAudioFrame) -> tuple[PcmChunk, ...]:
        """Convert one ordered SDK frame and return all newly complete packets."""
        self._stats = replace(
            self._stats,
            frames_received=self._stats.frames_received + 1,
            source_reported_drops=(self._stats.source_reported_drops + frame.source_dropped_frames),
        )
        if self._previous_sequence is not None:
            if frame.sequence <= self._previous_sequence:
                self._stats = replace(
                    self._stats,
                    frames_discarded_out_of_order=(self._stats.frames_discarded_out_of_order + 1),
                )
                return ()
            gap = frame.sequence - self._previous_sequence - 1
            if gap:
                self._stats = replace(
                    self._stats,
                    sequence_gap_frames=self._stats.sequence_gap_frames + gap,
                )
        self._previous_sequence = frame.sequence

        packets: list[PcmChunk] = []
        if self._source_format is not None and frame.format != self._source_format:
            packets.extend(self._finish_source_stream(drop_trailing=True))
            self._stats = replace(
                self._stats,
                format_changes=self._stats.format_changes + 1,
            )
        if self._source_format is None:
            self._start_source_stream(frame.format)

        mono = self._decode_mono(frame)
        resampled = self._resample(mono)
        packets.extend(self._packetize(resampled))
        return tuple(packets)

    def flush(self) -> tuple[PcmChunk, ...]:
        """Flush resampler delay and discard, rather than pad, a short final tail."""
        return tuple(self._finish_source_stream(drop_trailing=True))

    def _start_source_stream(self, source_format: VoiceAudioFormat) -> None:
        if source_format.sample_format != "f32le":
            raise ValueError("Voice ingress requires f32le remote audio")
        self._source_format = source_format
        self._resampler = (
            None
            if source_format.sample_rate == PROVIDER_INPUT_FORMAT.sample_rate
            else soxr.ResampleStream(
                source_format.sample_rate,
                PROVIDER_INPUT_FORMAT.sample_rate,
                1,
                dtype="float32",
                quality=self._quality,
            )
        )

    def _finish_source_stream(self, *, drop_trailing: bool) -> list[PcmChunk]:
        if self._source_format is None:
            return []
        packets: list[PcmChunk] = []
        if self._resampler is not None:
            tail = self._resampler.resample_chunk(
                np.empty(0, dtype=np.float32),
                last=True,
            )
            packets.extend(self._packetize(tail))
        if drop_trailing and self._pending_pcm:
            trailing_samples = len(self._pending_pcm) // PROVIDER_INPUT_FORMAT.frame_width_bytes
            self._stats = replace(
                self._stats,
                trailing_samples_dropped=(self._stats.trailing_samples_dropped + trailing_samples),
            )
            self._pending_pcm.clear()
        self._source_format = None
        self._resampler = None
        return packets

    def _decode_mono(self, frame: RemoteAudioFrame) -> np.ndarray:
        samples = np.frombuffer(frame.pcm, dtype="<f4")
        if frame.format.channels > 1:
            samples = samples.reshape(-1, frame.format.channels).mean(axis=1, dtype=np.float32)
        else:
            samples = samples.astype(np.float32, copy=False)
        invalid = int(np.count_nonzero(~np.isfinite(samples)))
        if invalid:
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
            self._stats = replace(
                self._stats,
                invalid_samples=self._stats.invalid_samples + invalid,
            )
        return samples

    def _resample(self, mono: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return mono
        return self._resampler.resample_chunk(mono)

    def _packetize(self, samples: np.ndarray) -> list[PcmChunk]:
        if samples.size:
            clipped_count = int(np.count_nonzero((samples < -1.0) | (samples > 1.0)))
            clipped = np.clip(samples, -1.0, 1.0)
            integers = np.clip(
                np.rint(clipped * 32768.0),
                -32768,
                32767,
            ).astype("<i2")
            self._pending_pcm.extend(integers.tobytes())
            if clipped_count:
                self._stats = replace(
                    self._stats,
                    clipped_samples=self._stats.clipped_samples + clipped_count,
                )

        packets: list[PcmChunk] = []
        while len(self._pending_pcm) >= INPUT_PACKET_BYTES:
            pcm = bytes(self._pending_pcm[:INPUT_PACKET_BYTES])
            del self._pending_pcm[:INPUT_PACKET_BYTES]
            packets.append(
                PcmChunk(
                    pcm,
                    PROVIDER_INPUT_FORMAT,
                    INPUT_PACKET_DURATION_MS,
                    generation=0,
                )
            )
        if packets:
            self._stats = replace(
                self._stats,
                packets_emitted=self._stats.packets_emitted + len(packets),
            )
        return packets


@dataclass(frozen=True, slots=True)
class VoiceInputQueueStats:
    """Cumulative Provider-input queue counters."""

    enqueued: int = 0
    delivered: int = 0
    dropped_oldest: int = 0


class VoiceInputQueue:
    """Bound Provider input latency by dropping the oldest complete 20 ms packet."""

    _CLOSED = object()

    def __init__(self, max_duration_ms: int) -> None:
        if max_duration_ms < INPUT_PACKET_DURATION_MS:
            raise ValueError("Voice input queue must hold at least one 20 ms packet")
        self._max_chunks = max_duration_ms // INPUT_PACKET_DURATION_MS
        self._queue: asyncio.Queue[PcmChunk | object] = asyncio.Queue(self._max_chunks)
        self._closed = False
        self._stats = VoiceInputQueueStats()

    @property
    def max_chunks(self) -> int:
        return self._max_chunks

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> VoiceInputQueueStats:
        return self._stats

    def put(self, chunk: PcmChunk) -> bool:
        """Enqueue current audio and report whether an older packet was dropped."""
        if self._closed:
            raise VoiceAudioQueueClosedError
        if chunk.format != PROVIDER_INPUT_FORMAT or chunk.duration_ms != INPUT_PACKET_DURATION_MS:
            raise ValueError("Voice input queue accepts exact 20 ms Provider input packets")
        dropped = False
        if self._queue.full():
            self._queue.get_nowait()
            dropped = True
        self._queue.put_nowait(chunk)
        self._stats = replace(
            self._stats,
            enqueued=self._stats.enqueued + 1,
            dropped_oldest=self._stats.dropped_oldest + int(dropped),
        )
        return dropped

    async def get(self) -> PcmChunk:
        item = await self._queue.get()
        if item is self._CLOSED:
            self._queue.put_nowait(self._CLOSED)
            raise VoiceAudioQueueClosedError
        if not isinstance(item, PcmChunk):  # pragma: no cover - internal queue invariant
            raise RuntimeError("Voice input queue contained an unknown item")
        self._stats = replace(self._stats, delivered=self._stats.delivered + 1)
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(self._CLOSED)


@dataclass(frozen=True, slots=True)
class VoiceOutputQueueStats:
    """Cumulative output generation and backpressure counters."""

    enqueued: int = 0
    delivered: int = 0
    stale_generation_dropped: int = 0
    flushed_chunks: int = 0
    backpressure_timeouts: int = 0


class VoiceOutputTransitQueue:
    """Bound output memory, preserve order, and reject stale generations after flush."""

    def __init__(self, max_duration_ms: int, *, put_timeout_seconds: float = 1.0) -> None:
        if max_duration_ms <= 0 or put_timeout_seconds <= 0:
            raise ValueError("Voice output queue bounds must be positive")
        self._max_duration_ms = max_duration_ms
        self._put_timeout_seconds = put_timeout_seconds
        self._items: deque[PcmChunk] = deque()
        self._buffered_duration_ms = 0
        self._generation = 0
        self._condition = asyncio.Condition()
        self._closed = False
        self._stats = VoiceOutputQueueStats()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def qsize(self) -> int:
        return len(self._items)

    @property
    def buffered_duration_ms(self) -> int:
        return self._buffered_duration_ms

    @property
    def max_chunk_duration_ms(self) -> int:
        return self._max_duration_ms

    @property
    def stats(self) -> VoiceOutputQueueStats:
        return self._stats

    async def start_generation(self) -> int:
        """Advance generation and atomically discard queued audio from the old response."""
        async with self._condition:
            if self._closed:
                raise VoiceAudioQueueClosedError
            self._generation += 1
            self._discard_all_locked()
            self._condition.notify_all()
            return self._generation

    async def put(self, chunk: PcmChunk) -> bool:
        """Backpressure matching audio or quickly discard a stale generation."""
        if chunk.format != PROVIDER_OUTPUT_FORMAT:
            raise ValueError("Voice output queue requires mono s16le 24 kHz PCM")
        if chunk.duration_ms > self._max_duration_ms:
            raise ValueError("One output chunk cannot exceed the entire queue bound")
        try:
            async with asyncio.timeout(self._put_timeout_seconds):
                async with self._condition:
                    while True:
                        if self._closed:
                            raise VoiceAudioQueueClosedError
                        if chunk.generation != self._generation:
                            self._stats = replace(
                                self._stats,
                                stale_generation_dropped=(self._stats.stale_generation_dropped + 1),
                            )
                            return False
                        if self._buffered_duration_ms + chunk.duration_ms <= self._max_duration_ms:
                            self._items.append(chunk)
                            self._buffered_duration_ms += chunk.duration_ms
                            self._stats = replace(
                                self._stats,
                                enqueued=self._stats.enqueued + 1,
                            )
                            self._condition.notify_all()
                            return True
                        await self._condition.wait()
        except TimeoutError as exc:
            async with self._condition:
                self._stats = replace(
                    self._stats,
                    backpressure_timeouts=self._stats.backpressure_timeouts + 1,
                )
            raise VoiceOutputBackpressureError from exc

    async def get(self) -> PcmChunk:
        async with self._condition:
            while not self._items:
                if self._closed:
                    raise VoiceAudioQueueClosedError
                await self._condition.wait()
            chunk = self._items.popleft()
            self._buffered_duration_ms -= chunk.duration_ms
            self._stats = replace(self._stats, delivered=self._stats.delivered + 1)
            self._condition.notify_all()
            return chunk

    async def wait_empty(self, generation: int) -> None:
        """Wait until one generation has left the project transit queue or is invalidated."""
        async with self._condition:
            while (
                not self._closed
                and generation == self._generation
                and any(chunk.generation == generation for chunk in self._items)
            ):
                await self._condition.wait()

    async def flush(self) -> int:
        """Invalidate the old response and wake blocked producers and consumers."""
        return await self.start_generation()

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            self._discard_all_locked()
            self._condition.notify_all()

    def _discard_all_locked(self) -> None:
        dropped = len(self._items)
        self._items.clear()
        self._buffered_duration_ms = 0
        if dropped:
            self._stats = replace(
                self._stats,
                flushed_chunks=self._stats.flushed_chunks + dropped,
            )

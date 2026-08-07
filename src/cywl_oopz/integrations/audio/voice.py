"""Adapt Provider voice PCM and cursors to the canonical shared audio bus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from cywl_oopz.features.audio.converter import StreamingAudioConverter
from cywl_oopz.features.audio.ledger import SourceKey
from cywl_oopz.features.audio.models import (
    AUDIO_SAMPLE_RATE,
    AudioFormat,
    AudioSourceKind,
    PcmSampleFormat,
    SourcePlaybackCursor,
)
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.models import PcmChunk, PlaybackCursor


@dataclass(slots=True)
class _VoiceSourceSegment:
    key: SourceKey
    native_frames: int = 0
    canonical_frames: int = 0
    closed: bool = False


class VoicePcmSourceOutput:
    """Expose the legacy voice media cursor contract over a canonical VOICE source."""

    def __init__(self, bus: SharedAudioMixerBus) -> None:
        self._bus = bus
        self._source_id = uuid4()
        self._source_generation = 0
        self._provider_generation = 0
        self._accepted_native_frames = 0
        self._segments: list[_VoiceSourceSegment] = []
        self._active_segment: _VoiceSourceSegment | None = None
        self._converter = self._new_converter()
        self._operation_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def write(self, chunk: PcmChunk) -> PlaybackCursor:
        if chunk.format != PROVIDER_OUTPUT_FORMAT:
            raise ValueError("VOICE source requires mono s16le 24 kHz PCM")
        async with self._operation_lock:
            self._require_open()
            if chunk.generation != self._provider_generation:
                raise ValueError("Voice PCM belongs to a different Provider generation")
            segment = self._ensure_active_segment()
            native_frames = len(chunk.pcm) // chunk.format.frame_width_bytes
            self._accepted_native_frames += native_frames
            segment.native_frames += native_frames
            blocks = self._converter.push(chunk.pcm, generation=self._source_generation)
            segment.canonical_frames = self._converter.generation_output_frames
            cursors = await self._bus.write_voice(blocks)
            return self._cursor_from(cursors)

    async def flush(self) -> PlaybackCursor:
        async with self._operation_lock:
            self._require_open()
            discard = frozenset(segment.key for segment in self._segments)
            plan = await self._bus.flush_voice(discard)
            old_cursor = self._cursor_from(plan.source_cursors)
            self._provider_generation += 1
            self._accepted_native_frames = 0
            self._segments.clear()
            self._active_segment = None
            self._reset_converter()
            return old_cursor

    async def drain(self) -> PlaybackCursor:
        async with self._operation_lock:
            self._require_open()
            if self._active_segment is not None:
                blocks = self._converter.flush(generation=self._source_generation)
                self._active_segment.canonical_frames = self._converter.generation_output_frames
                self._active_segment.closed = True
                await self._bus.write_voice(blocks)
                self._active_segment = None
                self._reset_converter()
            cursors = await self._bus.drain()
            return self._cursor_from(cursors)

    async def current_cursor(self) -> PlaybackCursor:
        async with self._operation_lock:
            self._require_open()
            return self._cursor_from(await self._bus.observe())

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._bus.aclose()
            self._closed = True

    def _ensure_active_segment(self) -> _VoiceSourceSegment:
        if self._active_segment is None:
            key = SourceKey(
                self._source_id,
                AudioSourceKind.VOICE,
                self._source_generation,
            )
            self._active_segment = _VoiceSourceSegment(key)
            self._segments.append(self._active_segment)
        return self._active_segment

    def _reset_converter(self) -> None:
        self._source_generation += 1
        self._converter.reset(self._source_generation)

    def _new_converter(self) -> StreamingAudioConverter:
        return StreamingAudioConverter(
            self._source_id,
            AudioSourceKind.VOICE,
            AudioFormat(
                PROVIDER_OUTPUT_FORMAT.sample_rate,
                PROVIDER_OUTPUT_FORMAT.channels,
                PcmSampleFormat(PROVIDER_OUTPUT_FORMAT.sample_format),
            ),
            generation=self._source_generation,
        )

    def _cursor_from(
        self,
        cursors: dict[SourceKey, SourcePlaybackCursor],
    ) -> PlaybackCursor:
        rendered_native = 0
        for segment in self._segments:
            cursor = cursors.get(segment.key)
            if cursor is None:
                continue
            if segment.closed and cursor.rendered_frames >= segment.canonical_frames:
                segment_rendered = segment.native_frames
            else:
                segment_rendered = min(
                    segment.native_frames,
                    cursor.rendered_frames
                    * PROVIDER_OUTPUT_FORMAT.sample_rate
                    // AUDIO_SAMPLE_RATE,
                )
            rendered_native += segment_rendered
        rendered_native = min(rendered_native, self._accepted_native_frames)
        return PlaybackCursor(
            self._provider_generation,
            self._accepted_native_frames,
            rendered_native,
            self._accepted_native_frames - rendered_native,
            PROVIDER_OUTPUT_FORMAT.sample_rate,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("VOICE source output is closed")

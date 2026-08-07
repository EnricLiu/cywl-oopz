"""Deterministic master PCM output used by audio session tests."""

from __future__ import annotations

import asyncio

from cywl_oopz.features.audio.errors import AudioBackpressureError, AudioQueueClosedError
from cywl_oopz.features.audio.models import MASTER_AUDIO_FORMAT, MasterPlaybackCursor


class FakeMasterPcmOutput:
    """Model accepted/rendered cursors without equating a write with playout."""

    def __init__(self, *, max_buffer_frames: int) -> None:
        if max_buffer_frames <= 0:
            raise ValueError("Fake master buffer must be positive")
        self._max_buffer_frames = max_buffer_frames
        self._epoch = 0
        self._accepted_frames = 0
        self._rendered_frames = 0
        self._closed = False
        self._lock = asyncio.Lock()
        self.writes: list[bytes] = []
        self.flush_count = 0
        self.drain_count = 0

    @property
    def cursor(self) -> MasterPlaybackCursor:
        return MasterPlaybackCursor(
            self._epoch,
            self._accepted_frames,
            self._rendered_frames,
            self._accepted_frames - self._rendered_frames,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    async def write(self, pcm_s16le: bytes) -> MasterPlaybackCursor:
        frames = MASTER_AUDIO_FORMAT.frames_for_bytes(pcm_s16le)
        async with self._lock:
            self._require_open()
            if self.cursor.buffered_frames + frames > self._max_buffer_frames:
                raise AudioBackpressureError("Fake master PCM buffer is full")
            self.writes.append(bytes(pcm_s16le))
            self._accepted_frames += frames
            return self.cursor

    async def flush(self) -> MasterPlaybackCursor:
        async with self._lock:
            self._require_open()
            old_cursor = self.cursor
            self._epoch += 1
            self._accepted_frames = 0
            self._rendered_frames = 0
            self.flush_count += 1
            return old_cursor

    async def drain(self) -> MasterPlaybackCursor:
        async with self._lock:
            self._require_open()
            self._rendered_frames = self._accepted_frames
            self.drain_count += 1
            return self.cursor

    async def advance_rendered(self, frames: int) -> MasterPlaybackCursor:
        """Advance the fake renderer by a bounded number of accepted frames."""
        if frames < 0:
            raise ValueError("Rendered frame advance must not be negative")
        async with self._lock:
            self._require_open()
            self._rendered_frames = min(
                self._accepted_frames,
                self._rendered_frames + frames,
            )
            return self.cursor

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._accepted_frames = 0
            self._rendered_frames = 0

    def _require_open(self) -> None:
        if self._closed:
            raise AudioQueueClosedError("Fake master PCM output is closed")

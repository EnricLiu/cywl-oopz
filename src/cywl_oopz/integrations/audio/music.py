"""Drive one decoded music stream through the canonical MUSIC source lane."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import uuid4

from cywl_oopz.core.observability import exception_kind
from cywl_oopz.features.audio.errors import AudioBusFailedError
from cywl_oopz.features.audio.ledger import SourceKey
from cywl_oopz.features.audio.models import (
    AUDIO_SAMPLE_RATE,
    AudioBlock,
    AudioSourceKind,
    DecodedAudioBlock,
    SourcePlaybackCursor,
)
from cywl_oopz.features.audio.ports import AudioDecoder
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.music.models import MusicPlaybackEndReason, MusicPlaybackResult

logger = logging.getLogger(__name__)


class MusicPcmSourceOutput:
    """Assign source identity and generation to canonical decoder blocks."""

    def __init__(self, bus: SharedAudioMixerBus) -> None:
        self._bus = bus
        self._source_id = uuid4()
        self._generation = 0
        self._next_frame = 0
        self._accepted_frames = 0

    @property
    def generation(self) -> int:
        return self._generation

    async def write(self, decoded: DecodedAudioBlock) -> SourcePlaybackCursor:
        block = AudioBlock(
            self._source_id,
            AudioSourceKind.MUSIC,
            self._generation,
            self._next_frame,
            decoded.valid_frames,
            decoded.samples,
        )
        self._next_frame += decoded.valid_frames
        self._accepted_frames += decoded.valid_frames
        cursors = await self._bus.write_music((block,))
        return self._cursor_from(cursors)

    async def flush(self) -> SourcePlaybackCursor:
        key = self._key()
        if self._accepted_frames:
            plan = await self._bus.flush_source(frozenset({key}))
            cursor = self._cursor_from(plan.source_cursors)
        else:
            cursor = SourcePlaybackCursor(self._generation, 0, 0)
        self._generation += 1
        self._next_frame = 0
        self._accepted_frames = 0
        return cursor

    async def drain(self) -> SourcePlaybackCursor:
        return self._cursor_from(await self._bus.drain(self._key(), release_source=True))

    def _cursor_from(
        self,
        cursors: dict[SourceKey, SourcePlaybackCursor],
    ) -> SourcePlaybackCursor:
        return cursors.get(
            self._key(),
            SourcePlaybackCursor(
                self._generation,
                self._accepted_frames,
                0,
            ),
        )

    def _key(self) -> SourceKey:
        return SourceKey(self._source_id, AudioSourceKind.MUSIC, self._generation)


class FfmpegMusicPlayback:
    """One cancellable decoder/source driver implementing the music playback port."""

    def __init__(self, decoder: AudioDecoder, source: MusicPcmSourceOutput) -> None:
        self._decoder = decoder
        self._source = source
        self._operation_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._resume = asyncio.Event()
        self._resume.set()
        self._stop_requested = False
        self._paused = False
        self._rendered_frames = 0
        self._last_cursor = SourcePlaybackCursor(0, 0, 0)
        self._result = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run(), name="music-pcm-playback")

    @classmethod
    def from_bus(
        cls,
        decoder: AudioDecoder,
        bus: SharedAudioMixerBus,
    ) -> FfmpegMusicPlayback:
        return cls(decoder, MusicPcmSourceOutput(bus))

    @property
    def finished(self) -> bool:
        return self._result.done()

    async def wait_finished(self) -> MusicPlaybackResult:
        return await asyncio.shield(self._result)

    async def stop(self) -> None:
        async with self._control_lock:
            if self.finished:
                return
            self._stop_requested = True
            self._resume.set()
            try:
                async with self._operation_lock:
                    cursor = await self._source.flush()
                    self._record_segment(cursor)
            except Exception as exc:
                self._complete(MusicPlaybackEndReason.TRACK_ERROR, exc)
            await self._decoder.aclose()
        with suppress(asyncio.CancelledError):
            await asyncio.shield(self._task)

    async def pause(self) -> bool:
        async with self._control_lock:
            if self.finished or self._paused:
                return False
            self._paused = True
            self._resume.clear()
            try:
                async with self._operation_lock:
                    cursor = await self._source.flush()
                    self._record_segment(cursor)
            except Exception as exc:
                self._complete(MusicPlaybackEndReason.TRACK_ERROR, exc)
                await self._decoder.aclose()
                raise
            return True

    async def resume(self) -> bool:
        async with self._control_lock:
            if self.finished or not self._paused:
                return False
            self._paused = False
            self._resume.set()
            return True

    async def aclose(self) -> None:
        await self.stop()

    def _start_log_decoder_stderr(self) -> None:
        if (aiter := self._decoder.status_logs()) is not None:

            async def _log_decoder_stderr(logs: AsyncIterator[str]) -> None:
                async for line in logs:
                    logger.debug("Music ffmpeg stderr: %s", line)
                logger.debug("Music ffmpeg stderr closed")

            logger.debug("Starting music decoder stderr log task")
            asyncio.create_task(_log_decoder_stderr(aiter), name="music-decoder-logs")

    async def _run(self) -> None:
        try:
            self._start_log_decoder_stderr()
            async for block in self._decoder:
                await self._resume.wait()
                if self._stop_requested:
                    break
                async with self._operation_lock:
                    if self._stop_requested:
                        break
                    self._last_cursor = await self._source.write(block)
            if self._stop_requested:
                self._complete(MusicPlaybackEndReason.STOPPED)
            else:
                async with self._operation_lock:
                    self._last_cursor = await self._source.drain()
                self._record_segment(self._last_cursor)
                self._complete(MusicPlaybackEndReason.FINISHED)
        except asyncio.CancelledError:
            self._complete(MusicPlaybackEndReason.BACKEND_CLOSED)
            raise
        except Exception as exc:
            try:
                async with self._operation_lock:
                    cursor = await self._source.flush()
                    self._record_segment(cursor)
            except Exception as cleanup_error:
                logger.warning(
                    "Could not flush failed PCM music source: error=%s",
                    exception_kind(cleanup_error),
                )
            reason = (
                MusicPlaybackEndReason.BACKEND_CLOSED
                if isinstance(exc, AudioBusFailedError)
                else MusicPlaybackEndReason.TRACK_ERROR
            )
            self._complete(reason, exc)
        finally:
            with suppress(Exception):
                await self._decoder.aclose()

    def _record_segment(self, cursor: SourcePlaybackCursor) -> None:
        self._last_cursor = cursor
        self._rendered_frames += cursor.rendered_frames

    def _complete(
        self,
        reason: MusicPlaybackEndReason,
        error: BaseException | None = None,
    ) -> None:
        if self._result.done():
            return
        duration = self._rendered_frames / AUDIO_SAMPLE_RATE
        self._result.set_result(MusicPlaybackResult(reason, duration, error))

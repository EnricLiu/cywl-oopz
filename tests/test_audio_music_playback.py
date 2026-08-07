from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cywl_oopz.features.audio.errors import AudioBusFailedError
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    DecodedAudioBlock,
)
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.music.models import MusicPlaybackEndReason
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput
from cywl_oopz.integrations.audio.music import FfmpegMusicPlayback, MusicPcmSourceOutput


def decoded(value: float = 0.25) -> DecodedAudioBlock:
    samples = np.full(
        (AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS),
        value,
        dtype=np.float32,
    )
    return DecodedAudioBlock(AUDIO_BLOCK_FRAMES, samples)


class TupleDecoder:
    def __init__(self, blocks: tuple[DecodedAudioBlock, ...]) -> None:
        self._blocks = iter(blocks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> DecodedAudioBlock:
        try:
            return next(self._blocks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


class GatedDecoder:
    def __init__(self) -> None:
        self.items: asyncio.Queue[DecodedAudioBlock | None] = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> DecodedAudioBlock:
        item = await self.items.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.items.put(None)


class FailedDecoder:
    def __init__(self) -> None:
        self._emitted = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> DecodedAudioBlock:
        if not self._emitted:
            self._emitted = True
            return decoded()
        raise RuntimeError("fixture decoder failure")

    async def aclose(self) -> None:
        self.closed = True


class FailedBus(SharedAudioMixerBus):
    async def write_music(self, blocks):
        del blocks
        raise AudioBusFailedError("fixture shared backend failure")


def playback_fixture(decoder) -> tuple[FfmpegMusicPlayback, FakeMasterPcmOutput]:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 11)
    return FfmpegMusicPlayback.from_bus(decoder, bus), master


@pytest.mark.asyncio
async def test_pcm_music_playback_drains_natural_eof() -> None:
    decoder = TupleDecoder((decoded(), decoded()))
    playback, master = playback_fixture(decoder)

    result = await playback.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.FINISHED
    assert result.duration_seconds == pytest.approx(0.04)
    assert len(master.writes) == 2
    assert master.drain_count == 1
    assert decoder.closed is True


@pytest.mark.asyncio
async def test_music_source_drain_releases_terminal_cursor_history() -> None:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 11)
    source = MusicPcmSourceOutput(bus)

    await source.write(decoded())
    cursor = await source.drain()

    assert cursor.accepted_frames == cursor.rendered_frames == AUDIO_BLOCK_FRAMES
    assert (await bus.stats()).retained_source_count == 0
    await bus.aclose()


@pytest.mark.asyncio
async def test_pcm_music_pause_flushes_buffer_and_resume_continues_decoder() -> None:
    decoder = GatedDecoder()
    playback, master = playback_fixture(decoder)
    await decoder.items.put(decoded())
    for _ in range(100):
        if master.writes:
            break
        await asyncio.sleep(0)

    assert await playback.pause() is True
    assert await playback.pause() is False
    assert master.flush_count == 1
    await decoder.items.put(decoded(0.5))
    await asyncio.sleep(0)
    assert len(master.writes) == 1
    assert await playback.resume() is True
    await decoder.items.put(None)
    result = await playback.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.FINISHED
    assert len(master.writes) == 2
    assert result.duration_seconds == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_pcm_music_stop_unblocks_decoder_and_returns_typed_result() -> None:
    decoder = GatedDecoder()
    playback, master = playback_fixture(decoder)

    await playback.stop()
    result = await playback.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.STOPPED
    assert decoder.closed is True
    assert master.flush_count == 0


@pytest.mark.asyncio
async def test_pcm_music_decoder_error_flushes_tail_and_releases_source() -> None:
    decoder = FailedDecoder()
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 11)
    playback = FfmpegMusicPlayback.from_bus(decoder, bus)

    result = await playback.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.TRACK_ERROR
    assert isinstance(result.terminal_error, RuntimeError)
    assert master.flush_count == 1
    assert decoder.closed is True
    assert (await bus.stats()).retained_source_count == 0
    await bus.aclose()


@pytest.mark.asyncio
async def test_pcm_music_maps_shared_bus_failure_to_recoverable_end_reason() -> None:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
    bus = FailedBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 11)
    playback = FfmpegMusicPlayback.from_bus(TupleDecoder((decoded(),)), bus)

    result = await playback.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.BACKEND_CLOSED
    assert isinstance(result.terminal_error, AudioBusFailedError)
    await bus.aclose()

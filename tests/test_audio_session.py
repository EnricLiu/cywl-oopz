from __future__ import annotations

import asyncio
from uuid import UUID

import numpy as np
import pytest

from cywl_oopz.features.audio.errors import AudioBackpressureError, AudioBusFailedError
from cywl_oopz.features.audio.ledger import SourceKey
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AudioBlock,
    AudioSourceKind,
    VoiceParticipantKind,
)
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput

_MUSIC_ID = UUID("10000000-0000-0000-0000-000000000001")
_VOICE_ID = UUID("20000000-0000-0000-0000-000000000002")


def block(kind: AudioSourceKind, value: float, *, start_frame: int = 0) -> AudioBlock:
    return AudioBlock(
        _MUSIC_ID if kind is AudioSourceKind.MUSIC else _VOICE_ID,
        kind,
        0,
        start_frame,
        AUDIO_BLOCK_FRAMES,
        np.full((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), value, dtype=np.float32),
    )


def bus_fixture() -> tuple[SharedAudioMixerBus, FakeMasterPcmOutput]:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
    bus = SharedAudioMixerBus(
        master,
        max_buffer_frames=AUDIO_BLOCK_FRAMES * 11,
    )
    bus.update_participants(
        frozenset({VoiceParticipantKind.MUSIC, VoiceParticipantKind.CONVERSATION})
    )
    return bus, master


class StalledMaster(FakeMasterPcmOutput):
    def __init__(self) -> None:
        super().__init__(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
        self.write_entered = asyncio.Event()
        self.write_allowed = asyncio.Event()

    async def write(self, pcm_s16le: bytes):
        self.write_entered.set()
        await self.write_allowed.wait()
        return await super().write(pcm_s16le)


class FailedMaster(FakeMasterPcmOutput):
    def __init__(self) -> None:
        super().__init__(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)

    async def write(self, pcm_s16le: bytes):
        del pcm_s16le
        raise RuntimeError("fixture transport failure")


@pytest.mark.asyncio
async def test_shared_bus_coalesces_music_and_voice_into_one_master_block() -> None:
    bus, master = bus_fixture()

    music_cursors, voice_cursors = await asyncio.gather(
        bus.write_music((block(AudioSourceKind.MUSIC, 0.1),)),
        bus.write_voice((block(AudioSourceKind.VOICE, 0.2),)),
    )

    assert len(master.writes) == 1
    samples = np.frombuffer(master.writes[0], dtype="<i2").reshape(
        AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS
    )
    assert np.all(samples != 0)
    assert samples[0, 0] > int(0.1 * 32_767)
    assert music_cursors[SourceKey(_MUSIC_ID, AudioSourceKind.MUSIC, 0)].accepted_frames == 960
    assert voice_cursors[SourceKey(_VOICE_ID, AudioSourceKind.VOICE, 0)].accepted_frames == 960
    await bus.aclose()


@pytest.mark.asyncio
async def test_user_speech_reason_smoothly_ducks_music_on_shared_bus() -> None:
    bus, master = bus_fixture()
    await bus.write_music((block(AudioSourceKind.MUSIC, 0.25),))
    await bus.set_user_speaking(True)
    await bus.write_music(
        (
            block(AudioSourceKind.MUSIC, 0.25, start_frame=AUDIO_BLOCK_FRAMES),
            block(AudioSourceKind.MUSIC, 0.25, start_frame=AUDIO_BLOCK_FRAMES * 2),
        )
    )

    first = np.frombuffer(master.writes[0], dtype="<i2")
    ducked = np.frombuffer(master.writes[-1], dtype="<i2")
    assert abs(int(ducked[-1])) < abs(int(first[-1])) / 2
    await bus.set_user_speaking(False)
    await bus.aclose()


@pytest.mark.asyncio
async def test_voice_flush_replays_only_unrendered_music_without_double_counting() -> None:
    bus, master = bus_fixture()
    music_key = SourceKey(_MUSIC_ID, AudioSourceKind.MUSIC, 0)
    voice_key = SourceKey(_VOICE_ID, AudioSourceKind.VOICE, 0)
    await asyncio.gather(
        bus.write_music(
            (
                block(AudioSourceKind.MUSIC, 0.1),
                block(AudioSourceKind.MUSIC, 0.1, start_frame=AUDIO_BLOCK_FRAMES),
            )
        ),
        bus.write_voice(
            (
                block(AudioSourceKind.VOICE, 0.2),
                block(AudioSourceKind.VOICE, 0.2, start_frame=AUDIO_BLOCK_FRAMES),
            )
        ),
    )
    await master.advance_rendered(AUDIO_BLOCK_FRAMES // 2)

    plan = await bus.flush_voice(frozenset({voice_key}))

    assert master.flush_count == 1
    assert len(master.writes) == 4
    assert plan.source_cursors[music_key].accepted_frames == AUDIO_BLOCK_FRAMES * 2
    assert plan.source_cursors[music_key].rendered_frames == AUDIO_BLOCK_FRAMES // 2
    assert plan.source_cursors[voice_key].rendered_frames == AUDIO_BLOCK_FRAMES // 2
    assert len(plan.survivors) == 2
    assert plan.survivors[0].music is not None
    assert plan.survivors[0].music.source_start_frame == AUDIO_BLOCK_FRAMES // 2
    assert plan.survivors[0].voice is None
    first_replay = np.frombuffer(master.writes[2], dtype="<i2").reshape(
        AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS
    )
    assert np.all(first_replay[AUDIO_BLOCK_FRAMES // 2 :] != 0)

    observed = await bus.drain()
    assert observed[music_key].accepted_frames == AUDIO_BLOCK_FRAMES * 2
    assert observed[music_key].rendered_frames == AUDIO_BLOCK_FRAMES * 2
    assert observed[voice_key].rendered_frames == AUDIO_BLOCK_FRAMES // 2
    await bus.aclose()


@pytest.mark.asyncio
async def test_music_flush_replays_voice_tail_and_preserves_voice_cursor() -> None:
    bus, master = bus_fixture()
    music_key = SourceKey(_MUSIC_ID, AudioSourceKind.MUSIC, 0)
    voice_key = SourceKey(_VOICE_ID, AudioSourceKind.VOICE, 0)
    await asyncio.gather(
        bus.write_music((block(AudioSourceKind.MUSIC, 0.1),)),
        bus.write_voice((block(AudioSourceKind.VOICE, 0.2),)),
    )

    plan = await bus.flush_source(frozenset({music_key}))
    observed = await bus.drain()

    assert len(plan.survivors) == 1
    assert plan.survivors[0].music is None
    assert plan.survivors[0].voice is not None
    assert observed[voice_key].accepted_frames == AUDIO_BLOCK_FRAMES
    assert observed[voice_key].rendered_frames == AUDIO_BLOCK_FRAMES
    assert observed[music_key].rendered_frames == 0
    await bus.aclose()


@pytest.mark.asyncio
async def test_one_hundred_barge_in_remixes_do_not_duplicate_music_cursor() -> None:
    bus, master = bus_fixture()
    music_key = SourceKey(_MUSIC_ID, AudioSourceKind.MUSIC, 0)

    for generation in range(100):
        voice_key = SourceKey(_VOICE_ID, AudioSourceKind.VOICE, generation)
        await asyncio.gather(
            bus.write_music(
                (
                    block(
                        AudioSourceKind.MUSIC,
                        0.1,
                        start_frame=generation * AUDIO_BLOCK_FRAMES,
                    ),
                )
            ),
            bus.write_voice(
                (
                    AudioBlock(
                        _VOICE_ID,
                        AudioSourceKind.VOICE,
                        generation,
                        0,
                        AUDIO_BLOCK_FRAMES,
                        np.full(
                            (AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS),
                            0.2,
                            dtype=np.float32,
                        ),
                    ),
                )
            ),
        )
        await master.advance_rendered(AUDIO_BLOCK_FRAMES // 2)
        await bus.flush_voice(frozenset({voice_key}))
        await master.advance_rendered(master.cursor.accepted_frames)
        await bus.observe()

    observed = await bus.drain()

    assert observed[music_key].accepted_frames == AUDIO_BLOCK_FRAMES * 100
    assert observed[music_key].rendered_frames == AUDIO_BLOCK_FRAMES * 100
    await bus.aclose()


@pytest.mark.asyncio
async def test_shared_bus_bounds_waiters_when_master_write_stalls() -> None:
    master = StalledMaster()
    bus = SharedAudioMixerBus(
        master,
        max_buffer_frames=AUDIO_BLOCK_FRAMES * 11,
        voice_queue_ms=20,
        put_timeout_seconds=0.01,
    )
    first = asyncio.create_task(bus.write_voice((block(AudioSourceKind.VOICE, 0.1),)))
    await master.write_entered.wait()
    queued = asyncio.create_task(bus.write_voice((block(AudioSourceKind.VOICE, 0.1),)))
    await asyncio.sleep(0)

    with pytest.raises(AudioBackpressureError, match="timed out"):
        await bus.write_voice((block(AudioSourceKind.VOICE, 0.1),))

    await bus.aclose()
    results = await asyncio.gather(first, queued, return_exceptions=True)
    assert all(isinstance(item, AudioBusFailedError) for item in results)
    assert master.closed is True


@pytest.mark.asyncio
async def test_master_transport_failure_fails_bus_and_future_writes() -> None:
    master = FailedMaster()
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 11)

    with pytest.raises(RuntimeError, match="transport failure"):
        await bus.write_music((block(AudioSourceKind.MUSIC, 0.1),))

    assert bus.failed is True
    with pytest.raises(AudioBusFailedError, match="failed"):
        await bus.write_music((block(AudioSourceKind.MUSIC, 0.1),))
    await bus.aclose()

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import numpy as np
import pytest

from cywl_oopz.features.audio.errors import (
    AudioBackpressureError,
    AudioLedgerCapacityError,
    AudioLedgerError,
    AudioQueueClosedError,
)
from cywl_oopz.features.audio.ledger import MasterPlayoutLedger, SourceKey
from cywl_oopz.features.audio.mixer import AudioMixer, MixedAudioBlock
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AudioBlock,
    AudioSourceKind,
    MasterPlaybackCursor,
    SourceSlice,
)
from cywl_oopz.features.audio.source_queue import AudioSourceQueue


def _block(
    source_id: UUID,
    kind: AudioSourceKind,
    *,
    generation: int = 0,
    start: int = 0,
    value: float = 0.1,
) -> AudioBlock:
    return AudioBlock(
        source_id,
        kind,
        generation,
        start,
        AUDIO_BLOCK_FRAMES,
        np.full((AUDIO_BLOCK_FRAMES, 2), value, dtype=np.float32),
    )


def _cursor(
    epoch: int,
    *,
    accepted: int,
    rendered: int,
) -> MasterPlaybackCursor:
    return MasterPlaybackCursor(epoch, accepted, rendered, accepted - rendered)


def _replay_mix(segment_music: SourceSlice) -> MixedAudioBlock:
    samples = np.zeros((AUDIO_BLOCK_FRAMES, 2), dtype=np.float32)
    start = segment_music.master_offset_frames
    end = start + segment_music.frame_count
    samples[start:end] = segment_music.samples
    samples.setflags(write=False)
    return MixedAudioBlock(
        samples,
        b"\0" * (AUDIO_BLOCK_FRAMES * 4),
        segment_music,
        None,
        AudioMixer().mix(None, None).limiter,
    )


def test_ledger_splits_partial_render_and_replays_only_music_after_voice_flush() -> None:
    music_id = uuid4()
    voice_id = uuid4()
    music = _block(music_id, AudioSourceKind.MUSIC)
    voice = _block(voice_id, AudioSourceKind.VOICE)
    ledger = MasterPlayoutLedger(max_buffer_frames=AUDIO_BLOCK_FRAMES * 4)
    entry = ledger.register_pending(AudioMixer().mix(music, voice))

    cursors = ledger.mark_accepted(
        entry,
        _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=AUDIO_BLOCK_FRAMES // 2),
    )
    music_key = SourceKey(music_id, AudioSourceKind.MUSIC, 0)
    voice_key = SourceKey(voice_id, AudioSourceKind.VOICE, 0)
    assert cursors[music_key].accepted_frames == AUDIO_BLOCK_FRAMES
    assert cursors[music_key].rendered_frames == AUDIO_BLOCK_FRAMES // 2
    assert cursors[voice_key].rendered_frames == AUDIO_BLOCK_FRAMES // 2

    plan = ledger.remix(
        _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=AUDIO_BLOCK_FRAMES // 2),
        discard=frozenset({voice_key}),
    )

    assert plan.old_epoch == 0 and plan.next_epoch == 1
    assert plan.last_rendered_sample is not None
    assert len(plan.survivors) == 1
    survivor = plan.survivors[0]
    assert survivor.voice is None and survivor.music is not None
    assert survivor.frame_count == AUDIO_BLOCK_FRAMES // 2
    assert survivor.music.source_start_frame == AUDIO_BLOCK_FRAMES // 2
    assert survivor.music.frame_count == AUDIO_BLOCK_FRAMES // 2


def test_replayed_source_range_does_not_increment_accepted_twice() -> None:
    music_id = uuid4()
    music = _block(music_id, AudioSourceKind.MUSIC)
    key = SourceKey(music_id, AudioSourceKind.MUSIC, 0)
    ledger = MasterPlayoutLedger(max_buffer_frames=AUDIO_BLOCK_FRAMES * 4)
    first = ledger.register_pending(AudioMixer().mix(music, None))
    ledger.mark_accepted(
        first,
        _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=AUDIO_BLOCK_FRAMES // 2),
    )
    plan = ledger.remix(_cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=AUDIO_BLOCK_FRAMES // 2))
    survivor = plan.survivors[0].music
    assert survivor is not None

    replay = ledger.register_pending(_replay_mix(survivor))
    cursors = ledger.mark_accepted(
        replay,
        _cursor(1, accepted=AUDIO_BLOCK_FRAMES, rendered=AUDIO_BLOCK_FRAMES // 2),
    )

    assert cursors[key].accepted_frames == AUDIO_BLOCK_FRAMES
    assert cursors[key].rendered_frames == AUDIO_BLOCK_FRAMES
    assert cursors[key].buffered_frames == 0


def test_flush_cursor_resolves_pending_write_acceptance_race() -> None:
    music_id = uuid4()
    voice_id = uuid4()
    ledger = MasterPlayoutLedger(max_buffer_frames=AUDIO_BLOCK_FRAMES * 2)
    pending = ledger.register_pending(
        AudioMixer().mix(
            _block(music_id, AudioSourceKind.MUSIC),
            _block(voice_id, AudioSourceKind.VOICE),
        )
    )
    assert ledger.pending_entry_id == pending

    plan = ledger.remix(
        _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=200),
        discard=frozenset({SourceKey(voice_id, AudioSourceKind.VOICE, 0)}),
    )

    assert ledger.pending_entry_id is None
    assert plan.survivors[0].music is not None
    assert plan.survivors[0].music.source_start_frame == 200
    assert plan.source_cursors[SourceKey(music_id, AudioSourceKind.MUSIC, 0)].rendered_frames == 200


def test_ledger_rejects_more_unrendered_audio_than_its_window() -> None:
    music_id = uuid4()
    ledger = MasterPlayoutLedger(max_buffer_frames=AUDIO_BLOCK_FRAMES)
    first = ledger.register_pending(AudioMixer().mix(_block(music_id, AudioSourceKind.MUSIC), None))
    ledger.mark_accepted(first, _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=0))

    with pytest.raises(AudioLedgerCapacityError):
        ledger.register_pending(
            AudioMixer().mix(
                _block(
                    music_id,
                    AudioSourceKind.MUSIC,
                    start=AUDIO_BLOCK_FRAMES,
                ),
                None,
            )
        )


def test_ledger_forgets_only_sources_without_master_references() -> None:
    music_id = uuid4()
    key = SourceKey(music_id, AudioSourceKind.MUSIC, 0)
    ledger = MasterPlayoutLedger(max_buffer_frames=AUDIO_BLOCK_FRAMES * 2)
    entry = ledger.register_pending(AudioMixer().mix(_block(music_id, AudioSourceKind.MUSIC), None))
    ledger.mark_accepted(
        entry,
        _cursor(0, accepted=AUDIO_BLOCK_FRAMES, rendered=0),
    )

    with pytest.raises(AudioLedgerError, match="still referenced"):
        ledger.forget_sources(frozenset({key}))

    cursors = ledger.observe(
        _cursor(
            0,
            accepted=AUDIO_BLOCK_FRAMES,
            rendered=AUDIO_BLOCK_FRAMES,
        )
    )
    assert cursors[key].rendered_frames == AUDIO_BLOCK_FRAMES
    assert ledger.entry_count == 0
    assert ledger.forget_sources(frozenset({key})) == 1
    assert ledger.source_count == 0
    assert ledger.forget_sources(frozenset({key})) == 0


@pytest.mark.asyncio
async def test_source_queue_flush_invalidates_blocked_old_generation() -> None:
    source_id = uuid4()
    queue = AudioSourceQueue(
        source_id,
        AudioSourceKind.VOICE,
        max_duration_ms=20,
    )
    assert await queue.put(_block(source_id, AudioSourceKind.VOICE)) is True
    blocked = asyncio.create_task(queue.put(_block(source_id, AudioSourceKind.VOICE, start=960)))
    await asyncio.sleep(0)
    assert blocked.done() is False

    flushed = await queue.start_generation(1)

    assert flushed == 1
    assert await blocked is False
    assert queue.qsize == 0
    assert queue.stats.flushed_blocks == 1
    assert queue.stats.stale_blocks == 1
    await queue.aclose()


@pytest.mark.asyncio
async def test_source_queue_backpressure_and_close_are_bounded() -> None:
    source_id = uuid4()
    queue = AudioSourceQueue(
        source_id,
        AudioSourceKind.MUSIC,
        max_duration_ms=20,
        put_timeout_seconds=0.01,
    )
    await queue.put(_block(source_id, AudioSourceKind.MUSIC))
    with pytest.raises(AudioBackpressureError):
        await queue.put(_block(source_id, AudioSourceKind.MUSIC, start=960))
    assert queue.stats.backpressure_timeouts == 1

    await queue.get()
    waiting = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    await queue.aclose()
    with pytest.raises(AudioQueueClosedError):
        await waiting
    with pytest.raises(AudioQueueClosedError):
        await queue.put(_block(source_id, AudioSourceKind.MUSIC))

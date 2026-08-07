from __future__ import annotations

import asyncio

import pytest

from cywl_oopz.features.audio.models import AUDIO_BLOCK_FRAMES, VoiceParticipantKind
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.models import PcmChunk
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput
from cywl_oopz.integrations.audio.voice import VoicePcmSourceOutput


def chunk(duration_ms: int, generation: int = 0, sample: int = 1000) -> PcmChunk:
    frames = PROVIDER_OUTPUT_FORMAT.sample_rate * duration_ms // 1_000
    pcm = sample.to_bytes(2, "little", signed=True) * frames
    return PcmChunk(pcm, PROVIDER_OUTPUT_FORMAT, duration_ms, generation)


def source_fixture() -> tuple[VoicePcmSourceOutput, FakeMasterPcmOutput]:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 20)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 21)
    return VoicePcmSourceOutput(bus), master


@pytest.mark.asyncio
async def test_voice_source_maps_master_drain_to_provider_rate_and_reuses_generation() -> None:
    source, master = source_fixture()

    accepted = await source.write(chunk(100))
    assert accepted.generation == 0
    assert accepted.accepted_samples == 2_400
    assert accepted.rendered_samples == 0
    assert master.writes
    assert all(len(pcm) == 3_840 for pcm in master.writes)

    first = await source.drain()
    assert first.accepted_samples == first.rendered_samples == 2_400
    await source.write(chunk(40))
    second = await source.drain()
    assert second.generation == 0
    assert second.accepted_samples == second.rendered_samples == 3_360
    await source.aclose()


@pytest.mark.asyncio
async def test_voice_source_flush_returns_old_cursor_and_rejects_stale_provider_audio() -> None:
    source, master = source_fixture()
    await source.write(chunk(100))
    await master.advance_rendered(AUDIO_BLOCK_FRAMES * 2)

    interrupted = await source.flush()

    assert interrupted.generation == 0
    assert interrupted.accepted_samples == 2_400
    assert interrupted.rendered_samples == 960
    assert master.flush_count == 1
    current = await source.current_cursor()
    assert current.generation == 1
    assert current.accepted_samples == current.rendered_samples == 0
    with pytest.raises(ValueError, match="different Provider generation"):
        await source.write(chunk(20, generation=0))

    await source.write(chunk(40, generation=1))
    drained = await source.drain()
    assert drained.generation == 1
    assert drained.accepted_samples == drained.rendered_samples == 960
    await source.aclose()


@pytest.mark.asyncio
async def test_voice_source_close_is_idempotent_and_blocks_further_output() -> None:
    source, master = source_fixture()

    await source.aclose()
    await source.aclose()

    assert master.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await source.write(chunk(20))


@pytest.mark.asyncio
async def test_empty_voice_flush_advances_generation_without_flushing_master() -> None:
    source, master = source_fixture()

    cursor = await source.flush()

    assert cursor.generation == 0
    assert cursor.accepted_samples == cursor.rendered_samples == 0
    assert master.flush_count == 0
    assert (await source.current_cursor()).generation == 1
    await source.aclose()


@pytest.mark.asyncio
async def test_voice_drain_uses_source_barrier_while_music_participant_is_active() -> None:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 20)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 21)
    bus.update_participants(
        frozenset({VoiceParticipantKind.MUSIC, VoiceParticipantKind.CONVERSATION})
    )
    source = VoicePcmSourceOutput(bus)
    await source.write(chunk(40))
    draining = asyncio.create_task(source.drain())
    for _ in range(100):
        if master.cursor.accepted_frames >= AUDIO_BLOCK_FRAMES * 2:
            break
        await asyncio.sleep(0.001)

    await master.advance_rendered(master.cursor.accepted_frames)
    cursor = await draining

    assert cursor.accepted_samples == cursor.rendered_samples == 960
    assert master.drain_count == 0
    await source.aclose()


@pytest.mark.asyncio
async def test_repeated_voice_drains_compact_rendered_source_segments() -> None:
    master = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 20)
    bus = SharedAudioMixerBus(master, max_buffer_frames=AUDIO_BLOCK_FRAMES * 21)
    source = VoicePcmSourceOutput(bus)

    for _ in range(100):
        await source.write(chunk(20))
        cursor = await source.drain()

    stats = await bus.stats()
    assert cursor.accepted_samples == cursor.rendered_samples == 48_000
    assert stats.retained_source_count == 0
    assert stats.ledger_entry_count == 0
    assert source._segments == []

    repeated = await source.drain()
    assert repeated == cursor
    assert master.drain_count == 100
    await source.aclose()

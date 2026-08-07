from __future__ import annotations

import pytest

from cywl_oopz.features.audio.errors import AudioBackpressureError, AudioQueueClosedError
from cywl_oopz.features.audio.models import AUDIO_BLOCK_FRAMES, MASTER_AUDIO_FORMAT
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput


def _pcm(frames: int = AUDIO_BLOCK_FRAMES) -> bytes:
    return b"\0" * (frames * MASTER_AUDIO_FORMAT.frame_width_bytes)


@pytest.mark.asyncio
async def test_fake_master_separates_accepted_from_rendered_and_flushes_epoch() -> None:
    output = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 2)

    first = await output.write(_pcm())
    second = await output.write(_pcm())

    assert first.accepted_frames == AUDIO_BLOCK_FRAMES
    assert first.rendered_frames == 0
    assert second.accepted_frames == AUDIO_BLOCK_FRAMES * 2
    assert second.buffered_frames == AUDIO_BLOCK_FRAMES * 2
    advanced = await output.advance_rendered(AUDIO_BLOCK_FRAMES // 2)
    assert advanced.rendered_frames == AUDIO_BLOCK_FRAMES // 2

    old = await output.flush()

    assert old.epoch == 0
    assert old.accepted_frames == AUDIO_BLOCK_FRAMES * 2
    assert old.rendered_frames == AUDIO_BLOCK_FRAMES // 2
    assert output.cursor.epoch == 1
    assert output.cursor.accepted_frames == output.cursor.rendered_frames == 0
    assert output.flush_count == 1


@pytest.mark.asyncio
async def test_fake_master_enforces_capacity_drain_and_close_contract() -> None:
    output = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES)
    await output.write(_pcm())
    with pytest.raises(AudioBackpressureError):
        await output.write(_pcm(1))

    drained = await output.drain()
    assert drained.accepted_frames == drained.rendered_frames == AUDIO_BLOCK_FRAMES
    assert drained.buffered_frames == 0
    assert output.drain_count == 1

    await output.aclose()
    await output.aclose()
    assert output.closed is True
    with pytest.raises(AudioQueueClosedError):
        await output.write(_pcm())

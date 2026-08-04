from __future__ import annotations

import pytest

from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceAudioFormat,
)


def test_voice_pcm_values_validate_alignment_and_cursor_order() -> None:
    input_format = VoiceAudioFormat(16_000, 1, "f32le")
    output_format = VoiceAudioFormat(24_000, 1, "s16le")

    frame = RemoteAudioFrame(b"\x00" * 64, input_format, 1, 12.5)
    chunk = PcmChunk(b"\x00" * 960, output_format, 20, 2)
    cursor = PlaybackCursor(2, 480, 240, 240, 24_000)

    assert frame.format.frame_width_bytes == 4
    assert chunk.format.frame_width_bytes == 2
    assert cursor.buffered_samples == 240


@pytest.mark.parametrize(
    "factory",
    (
        lambda: VoiceAudioFormat(0, 1, "s16le"),
        lambda: RemoteAudioFrame(b"\x00", VoiceAudioFormat(16_000, 1, "s16le"), 0, 0),
        lambda: PcmChunk(b"\x00\x00", VoiceAudioFormat(24_000, 1, "s16le"), 0, 0),
        lambda: PlaybackCursor(0, 1, 2, 0, 24_000),
    ),
)
def test_voice_pcm_values_reject_invalid_shapes(factory) -> None:
    with pytest.raises(ValueError):
        factory()

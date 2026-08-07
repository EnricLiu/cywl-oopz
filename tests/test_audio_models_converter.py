from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cywl_oopz.features.audio.converter import StreamingAudioConverter
from cywl_oopz.features.audio.errors import AudioFormatError
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AudioBlock,
    AudioFormat,
    AudioSourceKind,
    PcmSampleFormat,
)


def _valid_samples(blocks: tuple[AudioBlock, ...]) -> np.ndarray:
    return np.concatenate([block.samples[: block.valid_frames] for block in blocks], axis=0)


def _s16_mono(samples: np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def test_audio_block_copies_readonly_samples_and_requires_silent_tail() -> None:
    samples = np.zeros((AUDIO_BLOCK_FRAMES, 2), dtype=np.float32)
    samples[:4] = 0.25
    block = AudioBlock(
        uuid4(),
        AudioSourceKind.MUSIC,
        2,
        10,
        4,
        samples,
    )
    samples[0] = 0.75

    assert np.all(block.samples[0] == 0.25)
    assert block.samples.flags.writeable is False
    assert block.source_end_frame == 14
    with pytest.raises(ValueError):
        block.samples[0] = 0

    samples[8] = 0.5
    with pytest.raises(AudioFormatError, match="padding"):
        AudioBlock(
            uuid4(),
            AudioSourceKind.VOICE,
            0,
            0,
            4,
            samples,
        )


def test_source_slice_trim_maps_master_prefix_to_source_frames() -> None:
    samples = np.zeros((AUDIO_BLOCK_FRAMES, 2), dtype=np.float32)
    block = AudioBlock(uuid4(), AudioSourceKind.VOICE, 3, 1_000, 600, samples)
    source = block.as_slice(master_offset_frames=100)

    untouched = source.trim_master_prefix(50)
    assert untouched is not None
    assert untouched.source_start_frame == 1_000
    assert untouched.master_offset_frames == 50
    assert untouched.frame_count == 600

    trimmed = source.trim_master_prefix(300)
    assert trimmed is not None
    assert trimmed.source_start_frame == 1_200
    assert trimmed.master_offset_frames == 0
    assert trimmed.frame_count == 400
    assert source.trim_master_prefix(700) is None


def test_streaming_voice_converter_matches_single_and_awkward_input_chunks() -> None:
    input_format = AudioFormat(24_000, 1, PcmSampleFormat.S16LE)
    positions = np.arange(24_000, dtype=np.float32)
    tone = np.rint(12_000 * np.sin(2 * np.pi * 997 * positions / 24_000)).astype("<i2")
    source_id = uuid4()

    single_converter = StreamingAudioConverter(
        source_id,
        AudioSourceKind.VOICE,
        input_format,
    )
    single_blocks = single_converter.push(tone.tobytes(), generation=0)
    single_blocks += single_converter.flush(generation=0)

    split_converter = StreamingAudioConverter(
        source_id,
        AudioSourceKind.VOICE,
        input_format,
    )
    split_blocks: tuple[AudioBlock, ...] = ()
    offset = 0
    sizes = (137, 480, 911, 53, 1_024)
    index = 0
    while offset < tone.size:
        frame_count = sizes[index % len(sizes)]
        chunk = tone[offset : offset + frame_count]
        split_blocks += split_converter.push(chunk.tobytes(), generation=0)
        offset += chunk.size
        index += 1
    split_blocks += split_converter.flush(generation=0)

    single = _valid_samples(single_blocks)
    split = _valid_samples(split_blocks)
    assert single.shape == split.shape == (48_000, 2)
    np.testing.assert_array_equal(split, single)
    np.testing.assert_array_equal(split[:, 0], split[:, 1])
    assert [block.source_start_frame for block in split_blocks] == list(
        range(0, 48_000, AUDIO_BLOCK_FRAMES)
    )
    assert split_converter.native_frame_for_output(48_000) == 24_000
    assert split_converter.stats.emitted_blocks == 50


def test_converter_sanitizes_float_pcm_and_emits_zero_padded_tail() -> None:
    input_format = AudioFormat(48_000, 2, PcmSampleFormat.F32LE)
    converter = StreamingAudioConverter(
        uuid4(),
        AudioSourceKind.MUSIC,
        input_format,
    )
    samples = np.zeros((1_000, 2), dtype="<f4")
    samples[0] = [np.nan, np.inf]

    blocks = converter.push(samples.tobytes(), generation=0)
    blocks += converter.flush(generation=0)

    assert len(blocks) == 2
    assert blocks[0].valid_frames == AUDIO_BLOCK_FRAMES
    assert blocks[1].valid_frames == 40
    assert np.all(blocks[1].samples[40:] == 0)
    assert np.all(np.isfinite(_valid_samples(blocks)))
    assert converter.stats.invalid_samples == 2


def test_converter_reset_discards_old_filter_state_and_rejects_stale_generation() -> None:
    converter = StreamingAudioConverter(
        uuid4(),
        AudioSourceKind.VOICE,
        AudioFormat(24_000, 1, PcmSampleFormat.S16LE),
    )
    assert converter.push(_s16_mono(np.ones(100, dtype=np.int16)), generation=0) == ()

    discarded = converter.reset(1)

    assert discarded > 0
    with pytest.raises(ValueError, match="different"):
        converter.push(_s16_mono(np.ones(480, dtype=np.int16)), generation=0)
    blocks = converter.push(_s16_mono(np.zeros(480, dtype=np.int16)), generation=1)
    blocks += converter.flush(generation=1)
    assert blocks
    assert all(block.generation == 1 for block in blocks)
    assert blocks[0].source_start_frame == 0
    assert converter.stats.generation_resets == 1


def test_audio_format_rejects_empty_and_misaligned_pcm() -> None:
    format = AudioFormat(48_000, 2, PcmSampleFormat.S16LE)
    with pytest.raises(AudioFormatError):
        format.frames_for_bytes(b"")
    with pytest.raises(AudioFormatError):
        format.frames_for_bytes(b"\0\0")

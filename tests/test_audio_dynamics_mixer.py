from __future__ import annotations

import math
from uuid import uuid4

import numpy as np
import pytest

from cywl_oopz.features.audio.dynamics import (
    BlockPeakLimiter,
    GainEnvelope,
    db_to_linear,
    float32_stereo_to_s16le,
)
from cywl_oopz.features.audio.mixer import AudioMixer, DuckingSnapshot
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AudioBlock,
    AudioSourceKind,
    DuckingReason,
)


def _block(kind: AudioSourceKind, value: float, *, generation: int = 0) -> AudioBlock:
    return AudioBlock(
        uuid4(),
        kind,
        generation,
        0,
        AUDIO_BLOCK_FRAMES,
        np.full((AUDIO_BLOCK_FRAMES, 2), value, dtype=np.float32),
    )


def test_gain_envelope_is_continuous_across_two_twenty_ms_blocks() -> None:
    envelope = GainEnvelope(0.0)
    samples = np.ones((AUDIO_BLOCK_FRAMES, 2), dtype=np.float32)
    envelope.set_target_db(-6.0, duration_ms=40)

    first = envelope.process(samples)
    second = envelope.process(samples)

    assert first[0, 0] < 1.0
    assert first[-1, 0] > db_to_linear(-6.0)
    assert second[0, 0] < first[-1, 0]
    assert second[-1, 0] == pytest.approx(db_to_linear(-6.0), rel=1e-6)
    assert envelope.current_gain == pytest.approx(db_to_linear(-6.0), rel=1e-6)


def test_limiter_bounds_peak_sanitizes_invalid_and_releases_slowly() -> None:
    limiter = BlockPeakLimiter(-1.0, 120)
    loud = np.full((AUDIO_BLOCK_FRAMES, 2), 2.0, dtype=np.float32)
    loud[0] = [np.nan, np.inf]

    limited = limiter.process(loud)

    assert limited.invalid_samples == 2
    assert limited.output_peak <= db_to_linear(-1.0) + 1e-6
    assert limited.hard_clipped_samples == 0
    assert limited.gain_reduction_db > 6.0
    reduced_gain = limiter.gain

    limiter.process(np.zeros_like(loud))
    assert reduced_gain < limiter.gain < 1.0


def test_master_quantizer_maps_float_endpoints_without_overflow() -> None:
    samples = np.array([[-1.0, -0.5], [0.5, 1.0]], dtype=np.float32)
    quantized = np.frombuffer(float32_stereo_to_s16le(samples), dtype="<i2").reshape(-1, 2)

    np.testing.assert_array_equal(
        quantized,
        np.array([[-32768, -16384], [16384, 32767]], dtype=np.int16),
    )


def test_mixer_applies_solo_gain_then_smooth_ducking_and_retains_sources() -> None:
    mixer = AudioMixer()
    music = _block(AudioSourceKind.MUSIC, 0.5)

    solo = mixer.mix(music, None)
    first_duck = mixer.mix(
        music,
        None,
        DuckingSnapshot(True, frozenset({DuckingReason.USER_SPEECH})),
    )
    second_duck = mixer.mix(
        music,
        None,
        DuckingSnapshot(True, frozenset({DuckingReason.USER_SPEECH})),
    )

    assert solo.music is not None and solo.voice is None
    assert len(solo.pcm_s16le) == AUDIO_BLOCK_FRAMES * 2 * 2
    assert solo.samples[0, 0] == pytest.approx(0.5 * db_to_linear(-6.0), rel=1e-5)
    assert first_duck.samples[0, 0] < solo.samples[-1, 0]
    assert first_duck.samples[-1, 0] > second_duck.samples[-1, 0]
    assert second_duck.samples[-1, 0] == pytest.approx(
        0.5 * db_to_linear(-24.0),
        rel=1e-5,
    )


def test_mixer_limits_overlapping_sources_and_rejects_swapped_lanes() -> None:
    mixer = AudioMixer()
    music = _block(AudioSourceKind.MUSIC, 1.0)
    voice = _block(AudioSourceKind.VOICE, 1.0)

    mixed = mixer.mix(music, voice)

    assert mixed.music is not None and mixed.voice is not None
    assert float(np.max(np.abs(mixed.samples))) <= db_to_linear(-1.0) + 1e-6
    assert mixed.limiter.gain_reduction_db > 0
    assert math.isfinite(mixed.limiter.output_peak)
    with pytest.raises(ValueError, match="wrong"):
        mixer.mix(voice, music)

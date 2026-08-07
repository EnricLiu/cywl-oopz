"""Small deterministic gain, limiter, and PCM quantization primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE


def db_to_linear(db: float) -> float:
    if not math.isfinite(db):
        raise ValueError("Gain in dB must be finite")
    return 10.0 ** (db / 20.0)


class GainEnvelope:
    """Apply a continuous linear gain transition across arbitrary block boundaries."""

    def __init__(self, initial_db: float, *, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        if sample_rate <= 0:
            raise ValueError("Gain envelope sample rate must be positive")
        self._sample_rate = sample_rate
        self._current_gain = db_to_linear(initial_db)
        self._target_gain = self._current_gain
        self._remaining_frames = 0

    @property
    def current_gain(self) -> float:
        return self._current_gain

    @property
    def target_gain(self) -> float:
        return self._target_gain

    def set_target_db(self, target_db: float, *, duration_ms: int) -> bool:
        if duration_ms < 0:
            raise ValueError("Gain transition duration must not be negative")
        target = db_to_linear(target_db)
        if math.isclose(target, self._target_gain, rel_tol=0.0, abs_tol=1e-12):
            return False
        self._target_gain = target
        self._remaining_frames = self._sample_rate * duration_ms // 1_000
        if self._remaining_frames == 0:
            self._current_gain = target
        return True

    def process(self, samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != AUDIO_CHANNELS:
            raise ValueError("Gain envelope requires stereo frame arrays")
        frames = array.shape[0]
        if not frames:
            return np.empty_like(array)
        gains = np.empty(frames, dtype=np.float32)
        ramp_frames = min(frames, self._remaining_frames)
        if ramp_frames:
            ramp = np.linspace(
                self._current_gain,
                self._target_gain,
                self._remaining_frames + 1,
                dtype=np.float32,
            )[1 : ramp_frames + 1]
            gains[:ramp_frames] = ramp
            self._current_gain = float(ramp[-1])
            self._remaining_frames -= ramp_frames
        if ramp_frames < frames:
            gains[ramp_frames:] = self._target_gain
            self._current_gain = self._target_gain
            self._remaining_frames = 0
        return array * gains[:, np.newaxis]


@dataclass(frozen=True, slots=True)
class LimiterResult:
    samples: np.ndarray
    input_peak: float
    output_peak: float
    gain_reduction_db: float
    invalid_samples: int
    hard_clipped_samples: int


class BlockPeakLimiter:
    """Bound each known block peak and release attenuation across later blocks."""

    def __init__(
        self,
        threshold_db: float = -1.0,
        release_ms: int = 120,
        *,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        if threshold_db > 0 or release_ms <= 0 or sample_rate <= 0:
            raise ValueError("Limiter threshold/release/sample rate are invalid")
        self._threshold = db_to_linear(threshold_db)
        self._release_seconds = release_ms / 1_000
        self._sample_rate = sample_rate
        self._gain = 1.0

    @property
    def gain(self) -> float:
        return self._gain

    def reset(self) -> None:
        self._gain = 1.0

    def process(self, samples: np.ndarray) -> LimiterResult:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != AUDIO_CHANNELS or not array.shape[0]:
            raise ValueError("Limiter requires a non-empty stereo frame array")
        invalid = int(np.count_nonzero(~np.isfinite(array)))
        clean = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0, copy=True)
        input_peak = float(np.max(np.abs(clean)))
        required_gain = min(1.0, self._threshold / input_peak) if input_peak else 1.0
        if required_gain < self._gain:
            self._gain = required_gain
            limited = clean * self._gain
        else:
            release = math.exp(-array.shape[0] / (self._release_seconds * self._sample_rate))
            end_gain = min(required_gain, 1.0 - (1.0 - self._gain) * release)
            gains = np.linspace(self._gain, end_gain, array.shape[0], dtype=np.float32)
            limited = clean * gains[:, np.newaxis]
            self._gain = end_gain
        hard_clipped = int(np.count_nonzero((limited < -1.0) | (limited > 1.0)))
        limited = np.clip(limited, -1.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
        limited = np.ascontiguousarray(limited, dtype=np.float32)
        limited.setflags(write=False)
        output_peak = float(np.max(np.abs(limited)))
        reduction_db = -20.0 * math.log10(self._gain) if self._gain > 0 else math.inf
        return LimiterResult(
            limited,
            input_peak,
            output_peak,
            reduction_db,
            invalid,
            hard_clipped,
        )


def float32_stereo_to_s16le(samples: np.ndarray) -> bytes:
    """Quantize sanitized canonical samples into interleaved master PCM."""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != AUDIO_CHANNELS or not array.shape[0]:
        raise ValueError("Master quantizer requires non-empty stereo frames")
    clean = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    integers = np.clip(
        np.rint(np.clip(clean, -1.0, 1.0) * 32768.0),
        -32768,
        32767,
    ).astype("<i2")
    return integers.tobytes()

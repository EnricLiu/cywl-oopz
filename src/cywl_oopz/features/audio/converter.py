"""Streaming PCM conversion into canonical 48 kHz stereo float blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

import numpy as np
import soxr

from .errors import AudioFormatError
from .models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AudioBlock,
    AudioFormat,
    AudioSourceKind,
    PcmSampleFormat,
)


@dataclass(frozen=True, slots=True)
class AudioConverterStats:
    """Cumulative conversion counters across source generations."""

    input_frames: int = 0
    output_frames: int = 0
    emitted_blocks: int = 0
    invalid_samples: int = 0
    discarded_pending_frames: int = 0
    generation_resets: int = 0


class StreamingAudioConverter:
    """Keep resampler state while emitting exact canonical 20 ms blocks."""

    def __init__(
        self,
        source_id: UUID,
        kind: AudioSourceKind,
        input_format: AudioFormat,
        *,
        generation: int = 0,
        resample_quality: str = "HQ",
    ) -> None:
        if generation < 0:
            raise ValueError("Audio converter generation must be non-negative")
        self.source_id = source_id
        self.kind = kind
        self.input_format = input_format
        self._generation = generation
        self._quality = resample_quality
        self._resampler = self._new_resampler()
        self._pending = np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
        self._source_output_frames = 0
        self._source_input_frames = 0
        self._flushed = False
        self._stats = AudioConverterStats()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def generation_input_frames(self) -> int:
        return self._source_input_frames

    @property
    def generation_output_frames(self) -> int:
        return self._source_output_frames

    @property
    def stats(self) -> AudioConverterStats:
        return self._stats

    @property
    def algorithmic_delay_frames(self) -> int:
        if self._resampler is None:
            return 0
        return max(0, round(self._resampler.delay()))

    def push(self, pcm: bytes, *, generation: int) -> tuple[AudioBlock, ...]:
        """Convert aligned PCM bytes without finalizing the source generation."""
        self._require_generation(generation)
        if self._flushed:
            raise RuntimeError("Audio converter generation has already been flushed")
        native_frames = self.input_format.frames_for_bytes(pcm)
        decoded, invalid = self._decode(pcm)
        converted = self._convert(decoded, last=False)
        self._source_input_frames += native_frames
        self._stats = replace(
            self._stats,
            input_frames=self._stats.input_frames + native_frames,
            invalid_samples=self._stats.invalid_samples + invalid,
        )
        self._append(converted)
        return self._emit_full_blocks()

    def flush(self, *, generation: int) -> tuple[AudioBlock, ...]:
        """Flush resampler delay and return one zero-padded final block if needed."""
        self._require_generation(generation)
        if self._flushed:
            return ()
        self._flushed = True
        if self._resampler is not None:
            empty = np.empty((0, self.input_format.channels), dtype=np.float32)
            self._append(self._convert(empty, last=True))
        blocks = list(self._emit_full_blocks())
        if self._pending.shape[0]:
            valid_frames = self._pending.shape[0]
            padded = np.zeros((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), dtype=np.float32)
            padded[:valid_frames] = self._pending
            blocks.append(self._block(padded, valid_frames))
            self._pending = np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
        return tuple(blocks)

    def reset(self, generation: int) -> int:
        """Discard pending/filter state and begin a strictly newer source generation."""
        if generation <= self._generation:
            raise ValueError("Audio converter generation must increase")
        discarded = self._pending.shape[0] + self.algorithmic_delay_frames
        self._generation = generation
        self._pending = np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
        self._source_output_frames = 0
        self._source_input_frames = 0
        self._flushed = False
        self._resampler = self._new_resampler()
        self._stats = replace(
            self._stats,
            discarded_pending_frames=self._stats.discarded_pending_frames + discarded,
            generation_resets=self._stats.generation_resets + 1,
        )
        return discarded

    def native_frame_for_output(self, output_frame: int) -> int:
        """Map canonical progress to the input-rate timeline by duration."""
        if not 0 <= output_frame <= self._source_output_frames:
            raise ValueError("Canonical output frame is outside the current generation")
        native = output_frame * self.input_format.sample_rate // AUDIO_SAMPLE_RATE
        return min(native, self._source_input_frames)

    def _new_resampler(self) -> soxr.ResampleStream | None:
        if self.input_format.sample_rate == AUDIO_SAMPLE_RATE:
            return None
        return soxr.ResampleStream(
            self.input_format.sample_rate,
            AUDIO_SAMPLE_RATE,
            self.input_format.channels,
            dtype="float32",
            quality=self._quality,
        )

    def _decode(self, pcm: bytes) -> tuple[np.ndarray, int]:
        if self.input_format.sample_format is PcmSampleFormat.S16LE:
            decoded = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        elif self.input_format.sample_format is PcmSampleFormat.F32LE:
            decoded = np.frombuffer(pcm, dtype="<f4").astype(np.float32, copy=False)
        else:  # pragma: no cover - exhaustive enum guard
            raise AudioFormatError("Unsupported PCM sample format")
        decoded = decoded.reshape(-1, self.input_format.channels)
        invalid = int(np.count_nonzero(~np.isfinite(decoded)))
        if invalid:
            decoded = np.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=-1.0)
        return decoded, invalid

    def _convert(self, decoded: np.ndarray, *, last: bool) -> np.ndarray:
        converted = (
            decoded
            if self._resampler is None
            else self._resampler.resample_chunk(decoded, last=last)
        )
        converted = np.asarray(converted, dtype=np.float32)
        if converted.ndim == 1:
            converted = converted.reshape(-1, self.input_format.channels)
        if self.input_format.channels == 1:
            converted = np.repeat(converted, AUDIO_CHANNELS, axis=1)
        return np.ascontiguousarray(converted, dtype=np.float32)

    def _append(self, samples: np.ndarray) -> None:
        if not samples.shape[0]:
            return
        self._pending = np.concatenate((self._pending, samples), axis=0)

    def _emit_full_blocks(self) -> tuple[AudioBlock, ...]:
        blocks: list[AudioBlock] = []
        offset = 0
        while self._pending.shape[0] - offset >= AUDIO_BLOCK_FRAMES:
            samples = self._pending[offset : offset + AUDIO_BLOCK_FRAMES]
            blocks.append(self._block(samples, AUDIO_BLOCK_FRAMES))
            offset += AUDIO_BLOCK_FRAMES
        if offset:
            self._pending = np.ascontiguousarray(self._pending[offset:])
        return tuple(blocks)

    def _block(self, samples: np.ndarray, valid_frames: int) -> AudioBlock:
        block = AudioBlock(
            self.source_id,
            self.kind,
            self._generation,
            self._source_output_frames,
            valid_frames,
            samples,
        )
        self._source_output_frames += valid_frames
        self._stats = replace(
            self._stats,
            output_frames=self._stats.output_frames + valid_frames,
            emitted_blocks=self._stats.emitted_blocks + 1,
        )
        return block

    def _require_generation(self, generation: int) -> None:
        if generation != self._generation:
            raise ValueError("PCM belongs to a different audio source generation")

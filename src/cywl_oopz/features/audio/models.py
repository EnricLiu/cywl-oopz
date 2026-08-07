"""Provider- and SDK-neutral values for the shared PCM audio path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

import numpy as np

from .errors import AudioFormatError

AUDIO_BLOCK_DURATION_MS = 20
AUDIO_SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 2
AUDIO_BLOCK_FRAMES = AUDIO_SAMPLE_RATE * AUDIO_BLOCK_DURATION_MS // 1_000


@dataclass(frozen=True, slots=True)
class AudioChannelKey:
    """Stable OOPZ area/channel identity shared by music and conversation."""

    area_id: str
    channel_id: str

    def __post_init__(self) -> None:
        if not self.area_id.strip() or not self.channel_id.strip():
            raise ValueError("Audio channel area and channel identifiers must not be empty")


class VoiceParticipantKind(StrEnum):
    """Feature roles that may share one physical OOPZ voice channel."""

    MUSIC = "music"
    CONVERSATION = "conversation"


@dataclass(frozen=True, slots=True)
class VoiceParticipantRequest:
    """One idempotent feature owner requesting the shared voice session."""

    kind: VoiceParticipantKind
    channel: AudioChannelKey
    owner_key: str

    def __post_init__(self) -> None:
        if not self.owner_key.strip():
            raise ValueError("Voice participant owner key must not be empty")


class PcmSampleFormat(StrEnum):
    """PCM encodings accepted at source and master boundaries."""

    S16LE = "s16le"
    F32LE = "f32le"

    @property
    def sample_width_bytes(self) -> int:
        return 2 if self is PcmSampleFormat.S16LE else 4


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """One interleaved PCM format with an exact frame width."""

    sample_rate: int
    channels: int
    sample_format: PcmSampleFormat

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        if self.channels not in {1, 2}:
            raise ValueError("Audio core supports mono or stereo PCM")

    @property
    def frame_width_bytes(self) -> int:
        return self.channels * self.sample_format.sample_width_bytes

    def frames_for_bytes(self, pcm: bytes) -> int:
        if not pcm or len(pcm) % self.frame_width_bytes:
            raise AudioFormatError("PCM bytes must contain aligned non-empty frames")
        return len(pcm) // self.frame_width_bytes


CANONICAL_AUDIO_FORMAT = AudioFormat(
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    PcmSampleFormat.F32LE,
)
MASTER_AUDIO_FORMAT = AudioFormat(
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    PcmSampleFormat.S16LE,
)


class AudioSourceKind(StrEnum):
    """Fixed source lanes supported by the first mixer version."""

    MUSIC = "music"
    VOICE = "voice"


class DuckingReason(StrEnum):
    """Independent reasons that lower the music source gain."""

    USER_SPEECH = "user_speech"
    VOICE_PLAYOUT = "voice_playout"


def _readonly_audio_array(
    samples: np.ndarray,
    *,
    exact_frames: int | None = None,
) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim != 2 or array.shape[1] != AUDIO_CHANNELS:
        raise AudioFormatError("Audio samples must have shape (frames, 2)")
    if exact_frames is not None and array.shape[0] != exact_frames:
        raise AudioFormatError(f"Audio block must contain exactly {exact_frames} frames")
    if array.shape[0] <= 0:
        raise AudioFormatError("Audio samples must not be empty")
    if not np.issubdtype(array.dtype, np.floating):
        raise AudioFormatError("Canonical audio samples must be floating point")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AudioBlock:
    """One canonical 20 ms source block with optional zero-padded tail."""

    source_id: UUID
    kind: AudioSourceKind
    generation: int
    source_start_frame: int
    valid_frames: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if self.generation < 0 or self.source_start_frame < 0:
            raise ValueError("Audio source generation and frame position must be non-negative")
        if not 1 <= self.valid_frames <= AUDIO_BLOCK_FRAMES:
            raise ValueError("Audio block valid frame count is out of range")
        object.__setattr__(
            self,
            "samples",
            _readonly_audio_array(self.samples, exact_frames=AUDIO_BLOCK_FRAMES),
        )
        if self.valid_frames < AUDIO_BLOCK_FRAMES and np.any(self.samples[self.valid_frames :]):
            raise AudioFormatError("Audio block padding must be silent")

    @property
    def source_end_frame(self) -> int:
        return self.source_start_frame + self.valid_frames

    def as_slice(self, *, master_offset_frames: int = 0) -> SourceSlice:
        return SourceSlice(
            self.source_id,
            self.kind,
            self.generation,
            self.source_start_frame,
            master_offset_frames,
            self.samples[: self.valid_frames],
        )


@dataclass(frozen=True, slots=True)
class DecodedAudioBlock:
    """Canonical decoder output before a playback source assigns identity."""

    valid_frames: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if not 1 <= self.valid_frames <= AUDIO_BLOCK_FRAMES:
            raise ValueError("Decoded audio block valid frame count is out of range")
        object.__setattr__(
            self,
            "samples",
            _readonly_audio_array(self.samples, exact_frames=AUDIO_BLOCK_FRAMES),
        )
        if self.valid_frames < AUDIO_BLOCK_FRAMES and np.any(self.samples[self.valid_frames :]):
            raise AudioFormatError("Decoded audio block padding must be silent")


@dataclass(frozen=True, slots=True)
class SourceSlice:
    """A variable canonical source range aligned within one master segment."""

    source_id: UUID
    kind: AudioSourceKind
    generation: int
    source_start_frame: int
    master_offset_frames: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if min(self.generation, self.source_start_frame, self.master_offset_frames) < 0:
            raise ValueError("Source slice positions must be non-negative")
        object.__setattr__(self, "samples", _readonly_audio_array(self.samples))

    @property
    def frame_count(self) -> int:
        return self.samples.shape[0]

    @property
    def source_end_frame(self) -> int:
        return self.source_start_frame + self.frame_count

    @property
    def master_end_offset_frames(self) -> int:
        return self.master_offset_frames + self.frame_count

    def trim_master_prefix(self, prefix_frames: int) -> SourceSlice | None:
        """Remove master frames before one segment-local offset."""
        if prefix_frames < 0:
            raise ValueError("Source slice trim prefix must be non-negative")
        consumed = max(0, prefix_frames - self.master_offset_frames)
        if consumed >= self.frame_count:
            return None
        return SourceSlice(
            self.source_id,
            self.kind,
            self.generation,
            self.source_start_frame + consumed,
            max(0, self.master_offset_frames - prefix_frames),
            self.samples[consumed:],
        )


@dataclass(frozen=True, slots=True)
class MasterPlaybackCursor:
    """One SDK master generation cursor expressed in canonical frames."""

    epoch: int
    accepted_frames: int
    rendered_frames: int
    buffered_frames: int

    def __post_init__(self) -> None:
        if min(self.epoch, self.accepted_frames, self.rendered_frames, self.buffered_frames) < 0:
            raise ValueError("Master cursor values must be non-negative")
        if self.rendered_frames > self.accepted_frames:
            raise ValueError("Master rendered frames cannot exceed accepted frames")
        if self.rendered_frames + self.buffered_frames > self.accepted_frames:
            raise ValueError("Master buffered frames exceed accepted frames")

    @property
    def rendered_ms(self) -> int:
        return self.rendered_frames * 1_000 // AUDIO_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class SourcePlaybackCursor:
    """Monotonic source-local position independent from master epochs."""

    generation: int
    accepted_frames: int
    rendered_frames: int
    sample_rate: int = AUDIO_SAMPLE_RATE

    def __post_init__(self) -> None:
        if min(self.generation, self.accepted_frames, self.rendered_frames) < 0:
            raise ValueError("Source cursor values must be non-negative")
        if self.sample_rate <= 0 or self.rendered_frames > self.accepted_frames:
            raise ValueError("Source cursor frame counts or sample rate are invalid")

    @property
    def buffered_frames(self) -> int:
        return self.accepted_frames - self.rendered_frames

    @property
    def rendered_ms(self) -> int:
        return self.rendered_frames * 1_000 // self.sample_rate


@dataclass(frozen=True, slots=True)
class MixerLevels:
    """Finite source gain and transition settings used by one mixer."""

    music_solo_gain_db: float = -6.0
    music_voice_idle_gain_db: float = -10.0
    music_duck_gain_db: float = -24.0
    voice_gain_db: float = -3.0
    duck_attack_ms: int = 40
    duck_release_ms: int = 500
    limiter_threshold_db: float = -1.0
    limiter_release_ms: int = 120

    def __post_init__(self) -> None:
        gains = (
            self.music_solo_gain_db,
            self.music_voice_idle_gain_db,
            self.music_duck_gain_db,
            self.voice_gain_db,
            self.limiter_threshold_db,
        )
        if not all(math.isfinite(value) for value in gains):
            raise ValueError("Mixer gains must be finite")
        if self.limiter_threshold_db > 0:
            raise ValueError("Limiter threshold must not exceed 0 dBFS")
        if min(self.duck_attack_ms, self.duck_release_ms, self.limiter_release_ms) <= 0:
            raise ValueError("Mixer transition durations must be positive")

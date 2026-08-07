"""Fixed two-lane block mixer for music and realtime voice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import BlockPeakLimiter, GainEnvelope, LimiterResult, float32_stereo_to_s16le
from .models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AudioBlock,
    AudioSourceKind,
    DuckingReason,
    MixerLevels,
    SourceSlice,
)


@dataclass(frozen=True, slots=True)
class DuckingSnapshot:
    """One mixer tick's participant and speech activity."""

    conversation_active: bool = False
    reasons: frozenset[DuckingReason] = frozenset()


@dataclass(frozen=True, slots=True)
class MixedAudioBlock:
    """One quantized master block plus raw source contributions for the ledger."""

    samples: np.ndarray
    pcm_s16le: bytes
    music: SourceSlice | None
    voice: SourceSlice | None
    limiter: LimiterResult


class AudioMixer:
    """Mix fixed MUSIC and VOICE lanes with smooth ducking and peak limiting."""

    def __init__(self, levels: MixerLevels | None = None) -> None:
        self._levels = levels or MixerLevels()
        self._music_gain = GainEnvelope(self._levels.music_solo_gain_db)
        self._voice_gain = GainEnvelope(self._levels.voice_gain_db)
        self._limiter = BlockPeakLimiter(
            self._levels.limiter_threshold_db,
            self._levels.limiter_release_ms,
        )

    @property
    def limiter_gain(self) -> float:
        return self._limiter.gain

    def reset_dynamics(self) -> None:
        """Reset only master dynamics after an SDK generation flush."""
        self._limiter.reset()

    def mix(
        self,
        music: AudioBlock | None,
        voice: AudioBlock | None,
        ducking: DuckingSnapshot | None = None,
    ) -> MixedAudioBlock:
        self._validate_lane(music, AudioSourceKind.MUSIC)
        self._validate_lane(voice, AudioSourceKind.VOICE)
        snapshot = ducking or DuckingSnapshot()
        music_target = self._music_target(snapshot)
        duration = (
            self._levels.duck_attack_ms
            if self._music_gain.target_gain > 10.0 ** (music_target / 20.0)
            else self._levels.duck_release_ms
        )
        self._music_gain.set_target_db(music_target, duration_ms=duration)

        silence = np.zeros((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), dtype=np.float32)
        music_samples = music.samples if music is not None else silence
        voice_samples = voice.samples if voice is not None else silence
        mixed = self._music_gain.process(music_samples) + self._voice_gain.process(voice_samples)
        limited = self._limiter.process(mixed)
        return MixedAudioBlock(
            limited.samples,
            float32_stereo_to_s16le(limited.samples),
            music.as_slice() if music is not None else None,
            voice.as_slice() if voice is not None else None,
            limited,
        )

    def _music_target(self, snapshot: DuckingSnapshot) -> float:
        if snapshot.reasons:
            return self._levels.music_duck_gain_db
        if snapshot.conversation_active:
            return self._levels.music_voice_idle_gain_db
        return self._levels.music_solo_gain_db

    @staticmethod
    def _validate_lane(block: AudioBlock | None, expected: AudioSourceKind) -> None:
        if block is not None and block.kind is not expected:
            raise ValueError(f"{expected.value} mixer lane received the wrong source kind")

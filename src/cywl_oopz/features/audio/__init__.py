"""Shared realtime audio primitives for music and voice playout."""

from .models import (
    AUDIO_BLOCK_DURATION_MS,
    AUDIO_BLOCK_FRAMES,
    CANONICAL_AUDIO_FORMAT,
    MASTER_AUDIO_FORMAT,
    AudioBlock,
    AudioFormat,
    AudioSourceKind,
    DuckingReason,
    MasterPlaybackCursor,
    PcmSampleFormat,
    SourcePlaybackCursor,
    SourceSlice,
)

__all__ = [
    "AUDIO_BLOCK_DURATION_MS",
    "AUDIO_BLOCK_FRAMES",
    "CANONICAL_AUDIO_FORMAT",
    "MASTER_AUDIO_FORMAT",
    "AudioBlock",
    "AudioFormat",
    "AudioSourceKind",
    "DuckingReason",
    "MasterPlaybackCursor",
    "PcmSampleFormat",
    "SourcePlaybackCursor",
    "SourceSlice",
]

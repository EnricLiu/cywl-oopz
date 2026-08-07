"""Explicit offline duration gate for the shared audio state machine.

This test is CPU-only and never joins OOPZ. Enable it with
``CYWL_RUN_AUDIO_SOAK_TESTS=1``; it complements rather than replaces the real-room
30-minute gate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from uuid import UUID

import numpy as np
import pytest

from cywl_oopz.features.audio.dynamics import float32_stereo_to_s16le
from cywl_oopz.features.audio.ledger import SourceKey
from cywl_oopz.features.audio.mixer import AudioMixer, DuckingSnapshot
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_DURATION_MS,
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    MASTER_AUDIO_FORMAT,
    AudioBlock,
    AudioSourceKind,
    MasterPlaybackCursor,
    VoiceParticipantKind,
)
from cywl_oopz.features.audio.session import SharedAudioMixerBus

logger = logging.getLogger(__name__)

_MUSIC_ID = UUID("30000000-0000-0000-0000-000000000003")
_VOICE_ID = UUID("40000000-0000-0000-0000-000000000004")
_SOAK_MINUTES = 30
_BARGE_IN_COUNT = 100
_HEARTBEAT_INTERVAL_SECONDS = 0.005


class ImmediateMasterOutput:
    """Acknowledge and render immediately without retaining 30 minutes of PCM."""

    def __init__(self) -> None:
        self._epoch = 0
        self._frames = 0
        self.closed = False

    @property
    def cursor(self) -> MasterPlaybackCursor:
        return MasterPlaybackCursor(self._epoch, self._frames, self._frames, 0)

    async def write(self, pcm_s16le: bytes) -> MasterPlaybackCursor:
        self._frames += MASTER_AUDIO_FORMAT.frames_for_bytes(pcm_s16le)
        return self.cursor

    async def flush(self) -> MasterPlaybackCursor:
        old = self.cursor
        self._epoch += 1
        self._frames = 0
        return old

    async def drain(self) -> MasterPlaybackCursor:
        return self.cursor

    async def aclose(self) -> None:
        self.closed = True


def _enabled() -> bool:
    return os.environ.get("CYWL_RUN_AUDIO_SOAK_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _block(
    source_id: UUID,
    kind: AudioSourceKind,
    generation: int,
    start_frame: int,
    value: float,
) -> AudioBlock:
    return AudioBlock(
        source_id,
        kind,
        generation,
        start_frame,
        AUDIO_BLOCK_FRAMES,
        np.full((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), value, dtype=np.float32),
    )


async def _heartbeat(stop: asyncio.Event, lags: list[float]) -> None:
    previous = asyncio.get_running_loop().time()
    while not stop.is_set():
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        current = asyncio.get_running_loop().time()
        lags.append(max(0.0, current - previous - _HEARTBEAT_INTERVAL_SECONDS))
        previous = current


def _dsp_p99_ms(iterations: int = 5_000) -> float:
    mixer = AudioMixer()
    music = _block(_MUSIC_ID, AudioSourceKind.MUSIC, 0, 0, 0.1)
    voice = _block(_VOICE_ID, AudioSourceKind.VOICE, 0, 0, 0.2)
    snapshot = DuckingSnapshot(
        conversation_active=True,
        reasons=frozenset(),
    )
    durations_ns = np.empty(iterations, dtype=np.int64)
    for index in range(iterations):
        started_at = time.perf_counter_ns()
        mixed = mixer.mix(music, voice, snapshot)
        float32_stereo_to_s16le(mixed.samples)
        durations_ns[index] = time.perf_counter_ns() - started_at
    return float(np.percentile(durations_ns, 99) / 1_000_000)


@pytest.mark.asyncio
async def test_offline_thirty_minute_equivalent_mixing_is_bounded() -> None:
    if not _enabled():
        pytest.skip("set CYWL_RUN_AUDIO_SOAK_TESTS=1 for the explicit CPU soak")
    block_count = _SOAK_MINUTES * 60 * 1_000 // AUDIO_BLOCK_DURATION_MS
    remix_interval = block_count // _BARGE_IN_COUNT
    master = ImmediateMasterOutput()
    bus = SharedAudioMixerBus(
        master,
        max_buffer_frames=AUDIO_BLOCK_FRAMES * 9,
    )
    bus.update_participants(
        frozenset({VoiceParticipantKind.MUSIC, VoiceParticipantKind.CONVERSATION})
    )
    music_start = 0
    voice_start = 0
    voice_generation = 0
    started_at = time.monotonic()
    heartbeat_stop = asyncio.Event()
    heartbeat_lags: list[float] = []
    heartbeat_task = asyncio.create_task(
        _heartbeat(heartbeat_stop, heartbeat_lags),
        name="audio-soak-heartbeat",
    )

    try:
        for index in range(block_count):
            await asyncio.gather(
                bus.write_music(
                    (
                        _block(
                            _MUSIC_ID,
                            AudioSourceKind.MUSIC,
                            0,
                            music_start,
                            0.1,
                        ),
                    )
                ),
                bus.write_voice(
                    (
                        _block(
                            _VOICE_ID,
                            AudioSourceKind.VOICE,
                            voice_generation,
                            voice_start,
                            0.2,
                        ),
                    )
                ),
            )
            music_start += AUDIO_BLOCK_FRAMES
            voice_start += AUDIO_BLOCK_FRAMES
            if (index + 1) % remix_interval == 0:
                await bus.flush_voice(
                    frozenset(
                        {
                            SourceKey(
                                _VOICE_ID,
                                AudioSourceKind.VOICE,
                                voice_generation,
                            )
                        }
                    )
                )
                voice_generation += 1
                voice_start = 0
    finally:
        heartbeat_stop.set()
        await heartbeat_task

    await bus.forget_sources(frozenset({SourceKey(_MUSIC_ID, AudioSourceKind.MUSIC, 0)}))
    elapsed_seconds = time.monotonic() - started_at
    realtime_ratio = (_SOAK_MINUTES * 60) / elapsed_seconds
    heartbeat_max_lag_ms = max(heartbeat_lags, default=float("inf")) * 1_000
    dsp_p99_ms = _dsp_p99_ms()
    stats = await bus.stats()
    logger.info(
        "Offline shared audio soak: blocks=%s elapsed_s=%.3f realtime_ratio=%.1f "
        "heartbeat_ticks=%s heartbeat_max_lag_ms=%.3f dsp_p99_ms=%.3f metrics=%s",
        block_count,
        elapsed_seconds,
        realtime_ratio,
        len(heartbeat_lags),
        heartbeat_max_lag_ms,
        dsp_p99_ms,
        stats.as_metrics(),
    )

    assert stats.remix_count == _BARGE_IN_COUNT
    assert stats.retained_source_count == 0
    assert stats.ledger_entry_count == 0
    assert stats.hard_clip_samples == 0
    assert realtime_ratio >= 50
    assert len(heartbeat_lags) >= 10
    assert heartbeat_max_lag_ms < 100
    assert dsp_p99_ms < 2
    await bus.aclose()
    assert bus.closed is True
    assert master.closed is True

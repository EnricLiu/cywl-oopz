"""Opt-in Bilibili page -> yt-dlp -> FFmpeg canonical PCM gate."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.music.bilibili import BilibiliMusicProvider
from cywl_oopz.features.music.models import MusicSourceKind, MusicTrackReference
from cywl_oopz.integrations.audio.ffmpeg import (
    FfmpegMusicDecoder,
    FfmpegMusicDecoderFactory,
)
from cywl_oopz.integrations.media.ytdlp_runner import YtDlpProcessRunner
from cywl_oopz.settings import AudioMixerSettings, MusicSettings, YtDlpMusicSettings

logger = logging.getLogger(__name__)


def _live_enabled() -> bool:
    return os.getenv("CYWL_RUN_LIVE_BILIBILI_PCM_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.asyncio
async def test_live_bilibili_video_decodes_to_bounded_canonical_pcm() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_BILIBILI_PCM_TESTS=1 for real Bilibili PCM decode")

    music_values = dict(os.environ)
    music_values.update(
        {
            "CYWL_MUSIC_SOURCES": "bilibili",
            "CYWL_MUSIC_DEFAULT_SOURCE": "bilibili",
        }
    )
    music_settings = MusicSettings.from_mapping(music_values)
    ytdlp_settings = YtDlpMusicSettings.from_mapping(music_values)
    audio_settings = AudioMixerSettings.from_mapping(
        music_values | {"CYWL_AUDIO_MIXER_ENABLED": "true"}
    )
    source_id = os.getenv(
        "CYWL_MUSIC_LIVE_BILIBILI_SOURCE_ID",
        "BV13x41117TL:p=1",
    ).strip()
    runner = YtDlpProcessRunner(ytdlp_settings)
    provider = BilibiliMusicProvider(music_settings, ytdlp_settings, runner)
    decoder: FfmpegMusicDecoder | None = None
    try:
        track = await provider.lookup(MusicTrackReference(MusicSourceKind.BILIBILI, source_id))
        playable = await provider.resolve(track)
        factory = FfmpegMusicDecoderFactory(audio_settings)
        started_at = time.monotonic()
        decoder = await factory.open(playable.media)
        startup_ms = (time.monotonic() - started_at) * 1_000
        decoded_frames = 0
        block_count = 0
        async for block in decoder:
            assert block.samples.dtype == np.dtype("float32")
            assert block.samples.shape[1] == 2
            assert np.all(np.isfinite(block.samples[: block.valid_frames]))
            decoded_frames += block.valid_frames
            block_count += 1
            if decoded_frames >= 48_000:
                break

        logger.info(
            "Live Bilibili PCM gate: track=%s blocks=%s frames=%s startup_ms=%.1f",
            opaque_ref(playable.track.source_id),
            block_count,
            decoded_frames,
            startup_ms,
        )
        assert decoded_frames >= 48_000
        assert block_count >= 50
        assert decoder.stats.invalid_samples == 0
    finally:
        if decoder is not None:
            await decoder.aclose()
        await provider.aclose()
        await runner.aclose()

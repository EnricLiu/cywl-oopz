"""Explicit real Netease catalog -> FFmpeg PCM gate.

Start the configured NeteaseCloudMusicApi-compatible service, provide a working
FFmpeg binary, and set ``CYWL_RUN_LIVE_MUSIC_PCM_TESTS=1``. The test resolves one
temporary stream URL but never joins an OOPZ room.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.music.errors import MusicNotFoundError
from cywl_oopz.features.music.netease import NeteaseMusicCatalog
from cywl_oopz.integrations.audio.ffmpeg import FfmpegMusicDecoder, FfmpegMusicDecoderFactory
from cywl_oopz.settings import AudioMixerSettings, MusicSettings

logger = logging.getLogger(__name__)


def _live_enabled() -> bool:
    return os.getenv("CYWL_RUN_LIVE_MUSIC_PCM_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.asyncio
async def test_live_netease_stream_decodes_to_bounded_canonical_pcm() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_MUSIC_PCM_TESTS=1 for real Netease PCM decode")

    music_settings = MusicSettings.from_mapping(os.environ)
    audio_values = dict(os.environ)
    audio_values["CYWL_AUDIO_MIXER_ENABLED"] = "true"
    audio_settings = AudioMixerSettings.from_mapping(audio_values)
    catalog = NeteaseMusicCatalog(music_settings)
    decoder: FfmpegMusicDecoder | None = None
    try:
        query = os.getenv("CYWL_MUSIC_LIVE_QUERY", "初音未来").strip() or "初音未来"
        tracks = await catalog.search(query, limit=music_settings.search_limit)
        playable = None
        for track in tracks:
            try:
                playable = await catalog.resolve(track)
            except MusicNotFoundError:
                continue
            break
        if playable is None:
            pytest.fail("none of the bounded Netease search results is currently playable")

        factory = FfmpegMusicDecoderFactory(audio_settings)
        started_at = time.monotonic()
        decoder = await factory.open(playable.stream_url)
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
            "Live Netease PCM gate: track=%s blocks=%s frames=%s startup_ms=%.1f",
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
        await catalog.aclose()

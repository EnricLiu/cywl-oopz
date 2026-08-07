"""Explicit live shared-mixer gate for OOPZ/Agora realtime media.

Run only with ``CYWL_RUN_LIVE_VOICE_TESTS=1`` and the credential, area, channel,
and target-person variables documented by the SDK live voice guide. This test joins
the channel, mixes synthetic music with voice tones, and repeatedly flushes only the
voice lane. It consumes one owner frame for validation but never records remote PCM.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import time
from uuid import uuid4

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzBot, OopzConfig

from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AudioChannelKey,
    DecodedAudioBlock,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT, VoiceAudioIngress
from cywl_oopz.features.voice.models import (
    PcmChunk,
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.audio.music import MusicPcmSourceOutput
from cywl_oopz.integrations.oopz.master_audio import OopzMasterPcmOutputFactory
from cywl_oopz.integrations.oopz.voice_channel_session import (
    OopzVoiceChannelSessionManager,
)
from cywl_oopz.integrations.oopz.voice_media import OopzVoiceMediaGateway
from cywl_oopz.settings import AudioMixerSettings, VoiceSettings

logger = logging.getLogger(__name__)


def _live_enabled() -> bool:
    return os.environ.get("CYWL_RUN_LIVE_VOICE_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"live voice test requires {name}")
    return value


def _bounded_integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _tone(duration_ms: int = 100) -> PcmChunk:
    sample_count = PROVIDER_OUTPUT_FORMAT.sample_rate * duration_ms // 1000
    samples = (
        round(4_000 * math.sin(2 * math.pi * 440 * index / PROVIDER_OUTPUT_FORMAT.sample_rate))
        for index in range(sample_count)
    )
    return PcmChunk(
        struct.pack(f"<{sample_count}h", *samples),
        PROVIDER_OUTPUT_FORMAT,
        duration_ms,
        generation=0,
    )


def _music_block() -> DecodedAudioBlock:
    samples = np.full(
        (AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS),
        0.05,
        dtype=np.float32,
    )
    return DecodedAudioBlock(AUDIO_BLOCK_FRAMES, samples)


def test_live_voice_flush_count_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("CYWL_OOPZ_LIVE_FLUSH_COUNT", "100")

    assert _bounded_integer("CYWL_OOPZ_LIVE_FLUSH_COUNT", 100, minimum=1, maximum=100) == 100

    monkeypatch.setenv("CYWL_OOPZ_LIVE_FLUSH_COUNT", "101")
    with pytest.raises(ValueError, match="between 1 and 100"):
        _bounded_integer("CYWL_OOPZ_LIVE_FLUSH_COUNT", 100, minimum=1, maximum=100)


@pytest.mark.asyncio
async def test_live_project_shared_music_and_voice_pcm_survive_barge_in() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_VOICE_TESTS=1 for explicit RTC mutation")
    area_id = _required("OOPZ_AREA_ID")
    channel_id = _required("OOPZ_CHANNEL_ID")
    target_person_id = _required("OOPZ_TARGET_PERSON_UID")
    flush_count = _bounded_integer(
        "CYWL_OOPZ_LIVE_FLUSH_COUNT",
        100,
        minimum=1,
        maximum=100,
    )
    config = await OopzConfig.from_env_async()
    bot = OopzBot(config)
    audio_settings = AudioMixerSettings.from_mapping({"CYWL_AUDIO_MIXER_ENABLED": "true"})
    master_factory = OopzMasterPcmOutputFactory.from_settings(bot, audio_settings)
    sessions = OopzVoiceChannelSessionManager(
        bot,
        master_factory=master_factory,
        mixer_levels=audio_settings.mixer_levels(),
    )
    media = None
    conversation = None
    music = None
    try:
        await bot.rest.start()
        await bot.voice.start()
        channel = AudioChannelKey(area_id, channel_id)
        conversation = await sessions.try_acquire(
            VoiceParticipantRequest(
                VoiceParticipantKind.CONVERSATION,
                channel,
                owner_key="live-project-conversation",
            )
        )
        music = await sessions.try_acquire(
            VoiceParticipantRequest(
                VoiceParticipantKind.MUSIC,
                channel,
                owner_key="live-project-music",
            )
        )
        assert conversation is not None
        assert music is not None
        conversation_bus, music_bus = await asyncio.gather(
            conversation.audio_bus(),
            music.audio_bus(),
        )
        assert conversation_bus is music_bus
        music_output = MusicPcmSourceOutput(music_bus)
        settings = VoiceSettings.from_mapping(
            {
                "CYWL_VOICE_ENABLED": "true",
                "CYWL_VOICE_START_TIMEOUT_SECONDS": os.environ.get(
                    "CYWL_VOICE_START_TIMEOUT_SECONDS", "30"
                ),
            }
        )
        media = await OopzVoiceMediaGateway(
            bot,
            settings,
            audio_settings,
            master_factory=master_factory,
        ).open(
            VoiceSessionDescriptor(
                uuid4(),
                target_person_id,
                VoiceChannelKey(area_id, channel_id),
                VoiceTextAddress(area_id, channel_id),
            ),
            conversation,
        )

        input_iterator = media.input_frames()
        async with asyncio.timeout(settings.start_timeout_seconds):
            frame = await anext(input_iterator)
        ingress = VoiceAudioIngress()
        ingress.push(frame)
        assert ingress.stats.frames_received == 1
        assert frame.format.sample_format == "f32le"

        flush_latencies: list[float] = []
        previous_generation = -1
        for _ in range(flush_count):
            await asyncio.gather(
                music_output.write(_music_block()),
                media.write_output(_tone()),
            )
            started_at = time.monotonic()
            flushed = await media.flush_output()
            flush_latencies.append(time.monotonic() - started_at)
            assert flushed.generation > previous_generation
            current = await media.current_cursor()
            assert current.generation == flushed.generation + 1
            assert current.buffered_samples == 0
            previous_generation = flushed.generation
        p95_index = max(0, math.ceil(len(flush_latencies) * 0.95) - 1)
        flush_p95_seconds = sorted(flush_latencies)[p95_index]
        logger.info(
            "OOPZ project media live flush gate: count=%s p95_ms=%.1f max_ms=%.1f",
            flush_count,
            flush_p95_seconds * 1000,
            max(flush_latencies) * 1000,
        )
        assert flush_p95_seconds < 0.2

        music_cursor = await music_output.drain()
        assert music_cursor.accepted_frames == AUDIO_BLOCK_FRAMES * flush_count
        assert music_cursor.rendered_frames == music_cursor.accepted_frames
        audio_stats = await conversation_bus.stats()
        logger.info("OOPZ shared audio live gate: %s", audio_stats.as_metrics())
        assert audio_stats.remix_count == flush_count
        assert audio_stats.max_remix_ms < 200
        assert audio_stats.master_max_buffered_ms <= audio_settings.master_max_buffer_ms
        assert audio_stats.hard_clip_samples == 0
        assert audio_stats.retained_source_count == 0
        assert audio_stats.ledger_entry_count == 0

        snapshot = await sessions.current()
        assert snapshot is not None
        assert {item.kind for item in snapshot.participants} == {
            VoiceParticipantKind.MUSIC,
            VoiceParticipantKind.CONVERSATION,
        }
    finally:
        if media is not None:
            await media.aclose()
        if conversation is not None:
            await conversation.release()
        if music is not None:
            await music.release()
        await sessions.aclose()
        try:
            await bot.voice.close()
        finally:
            await bot.rest.close()

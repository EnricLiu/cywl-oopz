"""Explicit live project-adapter smoke for OOPZ/Agora realtime media.

Run only with ``CYWL_RUN_LIVE_VOICE_TESTS=1`` and the credential, area, channel,
and target-person variables documented by the SDK live voice guide. This test joins
the channel and emits a short 440 Hz tone; it never records remote PCM.
"""

from __future__ import annotations

import math
import os
import struct
from uuid import uuid4

import pytest
from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzBot, OopzConfig

from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT, VoiceAudioIngress
from cywl_oopz.features.voice.models import (
    PcmChunk,
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.oopz.voice_lease import (
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
    VoiceLeaseRequest,
)
from cywl_oopz.integrations.oopz.voice_media import OopzVoiceMediaGateway
from cywl_oopz.settings import VoiceSettings


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


@pytest.mark.asyncio
async def test_live_project_voice_media_receives_owner_and_plays_fixed_pcm() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_VOICE_TESTS=1 for explicit RTC mutation")
    area_id = _required("OOPZ_AREA_ID")
    channel_id = _required("OOPZ_CHANNEL_ID")
    target_person_id = _required("OOPZ_TARGET_PERSON_UID")
    config = await OopzConfig.from_env_async()
    bot = OopzBot(config)
    leases = OopzVoiceLeaseManager(bot)
    media = None
    lease = None
    try:
        await bot.rest.start()
        await bot.voice.start()
        lease = await leases.try_acquire(
            VoiceLeaseRequest(
                VoiceLeasePurpose.CONVERSATION,
                area_id,
                channel_id,
                owner_key="live-project-adapter",
            )
        )
        assert lease is not None
        settings = VoiceSettings.from_mapping(
            {
                "CYWL_VOICE_ENABLED": "true",
                "CYWL_VOICE_START_TIMEOUT_SECONDS": os.environ.get(
                    "CYWL_VOICE_START_TIMEOUT_SECONDS", "30"
                ),
            }
        )
        media = await OopzVoiceMediaGateway(bot, settings).open(
            VoiceSessionDescriptor(
                uuid4(),
                target_person_id,
                VoiceChannelKey(area_id, channel_id),
                VoiceTextAddress(area_id, channel_id),
            ),
            lease,
        )

        input_iterator = media.input_frames()
        frame = await anext(input_iterator)
        ingress = VoiceAudioIngress()
        ingress.push(frame)
        assert ingress.stats.frames_received == 1
        assert frame.format.sample_format == "f32le"

        await media.write_output(_tone())
        flushed = await media.flush_output()
        assert flushed.buffered_samples == 0
        await media.write_output(_tone())
        drained = await media.drain_output()
        assert drained.accepted_samples == drained.rendered_samples
    finally:
        if media is not None:
            await media.aclose()
        if lease is not None:
            await lease.release()
        await leases.aclose()
        try:
            await bot.voice.close()
        finally:
            await bot.rest.close()

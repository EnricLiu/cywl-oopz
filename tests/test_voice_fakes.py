from __future__ import annotations

import pytest

from cywl_oopz.features.voice.events import VoiceSessionReady
from cywl_oopz.features.voice.models import (
    PcmChunk,
    RemoteAudioFrame,
    VoiceAudioFormat,
)
from cywl_oopz.integrations.voice.fake import (
    FakeRealtimeVoiceSession,
    FakeVoiceMediaSession,
)


@pytest.mark.asyncio
async def test_fake_media_streams_input_and_tracks_output_cursor() -> None:
    media = FakeVoiceMediaSession()
    input_format = VoiceAudioFormat(16_000, 1, "f32le")
    output_format = VoiceAudioFormat(24_000, 1, "s16le")
    frame = RemoteAudioFrame(b"\x00" * 64, input_format, 0, 1.0)
    await media.push_input(frame)
    await media.end_input()

    assert [item async for item in media.input_frames()] == [frame]
    cursor = await media.write_output(PcmChunk(b"\x00" * 960, output_format, 20, 4))

    assert cursor.generation == 4
    assert cursor.accepted_samples == 480
    assert cursor.rendered_samples == 480
    assert (await media.flush_output()).generation == 5
    await media.aclose()


@pytest.mark.asyncio
async def test_fake_provider_session_yields_explicit_events_and_records_audio() -> None:
    session = FakeRealtimeVoiceSession()
    event = VoiceSessionReady()
    await session.emit(event)
    await session.aclose()

    assert [item async for item in session.events()] == [event]

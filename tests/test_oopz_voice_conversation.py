from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceSessionState,
    VoiceSessionStatus,
)
from cywl_oopz.integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from cywl_oopz.integrations.oopz.voice_conversation import (
    OopzConversationVoiceAccess,
    OopzVoiceCommandPresenter,
)


class FakeVoice:
    def __init__(self) -> None:
        self.joins: list[dict[str, str]] = []
        self.leaves = 0

    async def join(self, **values: str) -> None:
        self.joins.append(values)

    async def leave(self) -> None:
        self.leaves += 1


class FakeChannels:
    async def get_voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        assert (area_id, person_id) == ("area", "person")
        return "voice"


class FakeBot:
    def __init__(self) -> None:
        self.voice = FakeVoice()
        self.channels = FakeChannels()


class RecordingContext:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)


@pytest.mark.asyncio
async def test_oopz_conversation_access_maps_lookup_and_shared_lease_purpose() -> None:
    bot = FakeBot()
    leases = OopzVoiceChannelSessionManager(bot)
    access = OopzConversationVoiceAccess(bot, leases)

    channel_id = await access.voice_channel_for_user("area", "person")
    assert channel_id == "voice"
    lease = await access.try_acquire(VoiceChannelKey("area", channel_id), "session")

    assert lease is not None
    snapshot = await leases.current()
    assert snapshot is not None
    assert snapshot.participants[0].kind is VoiceParticipantKind.CONVERSATION
    assert bot.voice.joins == [{"area": "area", "channel": "voice"}]
    assert access.music_active(VoiceChannelKey("area", "voice")) is False

    music = await leases.try_acquire(
        VoiceParticipantRequest(
            VoiceParticipantKind.MUSIC,
            AudioChannelKey("area", "voice"),
            "music",
        )
    )
    assert music is not None
    assert access.music_active(VoiceChannelKey("area", "voice")) is True

    await lease.release()
    assert bot.voice.leaves == 0
    await music.release()
    assert bot.voice.leaves == 1
    await leases.aclose()


@pytest.mark.asyncio
async def test_voice_status_command_mentions_active_music_mix() -> None:
    context = RecordingContext()
    status = VoiceSessionStatus(
        active=True,
        session_id=uuid4(),
        owner_person_id="person",
        voice_channel=VoiceChannelKey("area", "voice"),
        state=VoiceSessionState.LISTENING,
        elapsed_seconds=12,
        metrics={"audio_music_participant_active": 1},
    )

    await OopzVoiceCommandPresenter().status(SimpleNamespace(responder=context), status)

    assert "与音乐混流中" in context.replies[0]

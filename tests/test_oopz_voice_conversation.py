from __future__ import annotations

import pytest

from cywl_oopz.features.voice.models import VoiceChannelKey
from cywl_oopz.integrations.oopz.voice_conversation import OopzConversationVoiceAccess
from cywl_oopz.integrations.oopz.voice_lease import (
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
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


@pytest.mark.asyncio
async def test_oopz_conversation_access_maps_lookup_and_shared_lease_purpose() -> None:
    bot = FakeBot()
    leases = OopzVoiceLeaseManager(bot)
    access = OopzConversationVoiceAccess(bot, leases)

    channel_id = await access.voice_channel_for_user("area", "person")
    assert channel_id == "voice"
    lease = await access.try_acquire(VoiceChannelKey("area", channel_id), "session")

    assert lease is not None
    snapshot = await leases.current()
    assert snapshot is not None
    assert snapshot.request.purpose is VoiceLeasePurpose.CONVERSATION
    assert bot.voice.joins == [{"area": "area", "channel": "voice"}]

    await lease.release()
    assert bot.voice.leaves == 1
    await leases.aclose()

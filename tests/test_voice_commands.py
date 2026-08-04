from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.voice.commands import VoiceCommand
from cywl_oopz.features.voice.service import VoiceConversationService
from cywl_oopz.integrations.oopz.voice_conversation import OopzVoiceCommandPresenter
from cywl_oopz.integrations.voice.fake import (
    FakeVoiceAccessGateway,
    FakeVoiceSessionRuntimeFactory,
)
from cywl_oopz.settings import VoiceSettings


@dataclass
class FakeMessage:
    plain_text: str
    sender_id: str = "person"
    area: str = "area"
    channel: str = "text"
    text: str = ""
    content: str = ""


@dataclass
class FakeContext:
    event: object
    replies: list[str] = field(default_factory=list)

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def context(message: FakeMessage, *, private: bool = False) -> FakeContext:
    return FakeContext(SimpleNamespace(message=message, is_private=private))


def command_fixture():
    voice_settings = VoiceSettings.from_mapping({"CYWL_VOICE_ENABLED": "true"})
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    service = VoiceConversationService(voice_settings, access, runtimes)
    router = CommandRouter("!")
    router.register(VoiceCommand(service, OopzVoiceCommandPresenter(), "!"))
    return router, service


@pytest.mark.asyncio
async def test_voice_command_start_status_and_stop_flow() -> None:
    router, service = command_fixture()

    start_message = FakeMessage("!voice start")
    start_context = context(start_message)
    await router.dispatch(start_message, start_context)
    assert "正在听" in start_context.replies[0]

    status_message = FakeMessage("!voice status")
    status_context = context(status_message)
    await router.dispatch(status_message, status_context)
    assert "正在听" in status_context.replies[0]
    assert "频道" in status_context.replies[0]

    stop_message = FakeMessage("!voice stop")
    stop_context = context(stop_message)
    await router.dispatch(stop_message, stop_context)
    assert "语音会话结束" in stop_context.replies[0]
    await service.aclose()


@pytest.mark.asyncio
async def test_voice_command_renders_usage_and_private_channel_error() -> None:
    router, service = command_fixture()
    invalid = FakeMessage("!voice model")
    invalid_context = context(invalid)
    await router.dispatch(invalid, invalid_context)
    assert "!voice start" in invalid_context.replies[0]

    private = FakeMessage("!voice start")
    private_context = context(private, private=True)
    await router.dispatch(private, private_context)
    assert "服务器文字频道" in private_context.replies[0]
    await service.aclose()

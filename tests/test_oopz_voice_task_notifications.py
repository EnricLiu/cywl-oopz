from __future__ import annotations

from types import SimpleNamespace

import pytest

from cywl_oopz.features.voice.models import VoiceTextAddress
from cywl_oopz.integrations.oopz.voice_task_notifications import OopzVoiceTaskTextGateway


class RecordingMessages:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def send_message(self, text: str, *, area: str, channel: str) -> None:
        self.calls.append((text, area, channel))


@pytest.mark.asyncio
async def test_voice_task_gateway_uses_the_origin_text_channel() -> None:
    messages = RecordingMessages()
    gateway = OopzVoiceTaskTextGateway(SimpleNamespace(messages=messages))

    await gateway.send(VoiceTextAddress("area-1", "text-2"), "✅ **后台任务 T1** 完成")

    assert messages.calls == [("✅ **后台任务 T1** 完成", "area-1", "text-2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " " * 10, "x" * 2001])
async def test_voice_task_gateway_rejects_invalid_oopz_message_lengths(text: str) -> None:
    gateway = OopzVoiceTaskTextGateway(SimpleNamespace(messages=RecordingMessages()))

    with pytest.raises(ValueError):
        await gateway.send(VoiceTextAddress("area", "text"), text)

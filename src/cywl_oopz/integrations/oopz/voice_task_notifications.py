"""OOPZ text-channel delivery for terminal delegated voice tasks."""

from __future__ import annotations

from typing import Any

from cywl_oopz.features.voice.models import VoiceTextAddress


class OopzVoiceTaskTextGateway:
    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def send(self, address: VoiceTextAddress, text: str) -> None:
        if not text.strip() or len(text) > 2000:
            raise ValueError("Voice task text notification must contain 1-2000 characters")
        await self._bot.messages.send_message(
            text,
            area=address.area_id,
            channel=address.channel_id,
        )
